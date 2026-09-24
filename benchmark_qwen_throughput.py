import os, re, json, time, csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

OPEN_TAG = "<" + "think>"
CLOSE_TAG = "</" + "think>"
MODEL_ID = "tianchuang/Qwen-3-8B-RHEA-property-predictor"

# 20 Diverse Ti-Alloys for testing
TEST_ALLOYS = [
    "Ti90Al6V4", "Ti85V15", "Ti90V9Fe1", "Ti90V9Cr1", "Ti90Al7V3",
    "Ti81Al9Mo2V5Cr3", "Ti84V9Fe2Al5", "Ti86Al10V4", "Ti90V8Mo2", "Ti90V9Zr1",
    "Ti85Al11V4", "Ti85V11Cr4", "Ti90Al2V8", "Ti90Al9Cr1", "Ti90V4Cr4Fe2",
    "Ti90Al8V1Cr1", "Ti83Al11Sn1Zr2Mo3", "Ti90Mo4Zr1Sn2Fe3", "Ti80Al8V2Zr3Cr7", "Ti75Al10V15"
]

BATCH_SIZES_TO_TEST = [1, 2, 4, 8]

def load_model():
    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", dtype=torch.bfloat16,
                                                 quantization_config=bnb, trust_remote_code=True)
    model.eval()
    return model, tokenizer

def build_prompt(tokenizer, formula):
    prompt = (f"Predict the compressive yield strength (in MPa), hardness (in HV), "
              f"density (in g/cm3), and primary phase for the alloy {formula} at 25C. "
              f'Output ONLY a valid JSON object with keys: "yield_strength_MPa", '
              f'"hardness_HV", "density_g_cm3", "phase".')
    messages = [{"role": "system", "content": "You are an expert metallurgist. Output only valid JSON."},
                {"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def parse_response(response):
    clean = re.compile(rf'{OPEN_TAG}.*?{CLOSE_TAG}', re.DOTALL).sub('', response).strip()
    data = {}
    m = re.search(r'\{.*\}', clean, re.DOTALL)
    if m:
        try: data = json.loads(m.group(0))
        except: pass
    if not data:
        for key, pat in [("yield_strength_MPa", r'yield[_\s]*strength[^0-9]*([0-9\.]+)'),
                         ("hardness_HV", r'hardness[^0-9]*([0-9\.]+)'),
                         ("density_g_cm3", r'density[^0-9]*([0-9\.]+)'),
                         ("phase", r'phase[^A-Za-z]*([A-Za-z0-9\-\+]+)')]:
            mm = re.search(pat, clean, re.IGNORECASE)
            if mm: data[key] = mm.group(1)
    return data, clean

def run_batch(model, tokenizer, formulas, batch_size):
    results = []
    for i in range(0, len(formulas), batch_size):
        chunk = formulas[i:i + batch_size]
        texts = [build_prompt(tokenizer, f) for f in chunk]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        
        n_in = inputs.input_ids.shape[1]
        responses = tokenizer.batch_decode(out[:, n_in:], skip_special_tokens=True)
        
        for f, resp in zip(chunk, responses):
            data, clean = parse_response(resp)
            results.append({"formula": f, "raw_text": clean, "parsed_data": data})
    return results

def main():
    model, tokenizer = load_model()
    
    throughput_results = []
    all_predictions = []
    
    for bs in BATCH_SIZES_TO_TEST:
        print(f"\n{'='*50}")
        print(f"TESTING BATCH SIZE: {bs}")
        print(f"{'='*50}")
        
        try:
            torch.cuda.synchronize()
            t0 = time.time()
            preds = run_batch(model, tokenizer, TEST_ALLOYS, bs)
            torch.cuda.synchronize()
            dt = time.time() - t0
            
            sps = len(TEST_ALLOYS) / dt
            print(f"Success! {len(TEST_ALLOYS)} alloys in {dt:.2f}s -> {sps:.3f} samples/sec")
            
            throughput_results.append({
                "batch_size": bs, "total_alloys": len(TEST_ALLOYS), 
                "time_sec": round(dt, 2), "samples_per_sec": round(sps, 3), "status": "SUCCESS"
            })
            
            # Save predictions to verify determinism
            for p in preds:
                p["batch_size_used"] = bs
                all_predictions.append(p)
                
        except torch.cuda.OutOfMemoryError:
            print(f"OOM ERROR at batch size {bs}! VRAM limit reached.")
            throughput_results.append({
                "batch_size": bs, "total_alloys": len(TEST_ALLOYS), 
                "time_sec": -1, "samples_per_sec": 0, "status": "OOM_FAILURE"
            })
            torch.cuda.empty_cache()
            # Don't test larger batch sizes if we OOM'd
            break 

    # Save Throughput Metrics
    with open("qwen_throughput_benchmark.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["batch_size", "total_alloys", "time_sec", "samples_per_sec", "status"])
        writer.writeheader()
        writer.writerows(throughput_results)
    print("\n[Saved] qwen_throughput_benchmark.csv")

    # Save Detailed Predictions (for reliability checking)
    with open("qwen_predictions_detailed.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["batch_size_used", "formula", "yield_strength_MPa", "hardness_HV", "density_g_cm3", "phase", "raw_text"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in all_predictions:
            row = {
                "batch_size_used": p["batch_size_used"],
                "formula": p["formula"],
                "yield_strength_MPa": p["parsed_data"].get("yield_strength_MPa", "PARSE_FAIL"),
                "hardness_HV": p["parsed_data"].get("hardness_HV", "PARSE_FAIL"),
                "density_g_cm3": p["parsed_data"].get("density_g_cm3", "PARSE_FAIL"),
                "phase": p["parsed_data"].get("phase", "PARSE_FAIL"),
                "raw_text": p["raw_text"].replace("\n", " ") # flatten for CSV
            }
            writer.writerow(row)
    print("[Saved] qwen_predictions_detailed.csv")

if __name__ == "__main__":
    main()
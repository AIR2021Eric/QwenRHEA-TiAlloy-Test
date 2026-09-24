"""
test_qwen_rhea.py (v3)
- 4-bit NF4 quantization (fits 8 GB VRAM)
- Batched generation (left-padding); batch size KV-cache-bound (<=4 on 8 GB)
- Optional thinking-trace disable for ~5x faster JSON-only answers
- Robust think-block stripping + JSON/regex parsing
- OOD guardrails: RHEA-extrapolation flag + Mo_eq phase-hallucination check
"""
import os, re, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# tags built by concatenation so editors/parsers never intercept them
OPEN_TAG = "<" + "think>"
CLOSE_TAG = "</" + "think>"

MODEL_ID = "tianchuang/Qwen-3-8B-RHEA-property-predictor"
BATCH_SIZE = 2            # KV-cache bound on 8 GB VRAM; try 4 if stable
ENABLE_THINKING = False   # False = skip reasoning traces -> short, fast JSON
MAX_NEW_TOKENS = 1500 if ENABLE_THINKING else 300

def estimate_kv_cache_mb(n_tokens, layers=36, kv_heads=8, head_dim=128, bytes_per=2):
    return 2 * layers * kv_heads * head_dim * bytes_per * n_tokens / 1e6

def load_model():
    print("=" * 60); print("LOADING QWEN-3-8B-RHEA (4-bit NF4)"); print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = "left"                      # required for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16,
        quantization_config=bnb, trust_remote_code=True)
    model.eval()
    kv = estimate_kv_cache_mb(MAX_NEW_TOKENS + 300)
    print(f"KV-cache estimate: ~{kv:.0f} MB per sequence -> "
          f"safe batch size on 8 GB VRAM: <= {max(1, int(2500 // kv))}")
    return model, tokenizer

def build_prompt(tokenizer, formula, temp_c=25):
    prompt = (f"Predict the compressive yield strength (in MPa), hardness (in HV), "
              f"density (in g/cm3), and primary phase for the alloy {formula} at {temp_c}C. "
              f'Output ONLY a valid JSON object with keys: "yield_strength_MPa", '
              f'"hardness_HV", "density_g_cm3", "phase".')
    messages = [{"role": "system",
                 "content": "You are an expert metallurgist. Output only valid JSON."},
                {"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=ENABLE_THINKING)
    except TypeError:   # template without thinking toggle
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)

def parse_response(response):
    clean = re.compile(rf'{OPEN_TAG}.*?{CLOSE_TAG}', re.DOTALL).sub('', response).strip()
    data = {}
    m = re.search(r'\{.*\}', clean, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
        except Exception:
            data = {}
    if not data:  # regex fallback
        for key, pat in [("yield_strength_MPa", r'yield[_\s]*strength[^0-9]*([0-9\.]+)'),
                         ("hardness_HV", r'hardness[^0-9]*([0-9\.]+)'),
                         ("density_g_cm3", r'density[^0-9]*([0-9\.]+)'),
                         ("phase", r'phase[^A-Za-z]*([A-Za-z0-9\-\+]+)')]:
            mm = re.search(pat, clean, re.IGNORECASE)
            if mm:
                data[key] = mm.group(1)
    return data, clean

def generate_texts(model, tokenizer, texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False, pad_token_id=tokenizer.pad_token_id)
    n_in = inputs.input_ids.shape[1]          # left padding -> uniform prompt length
    return tokenizer.batch_decode(out[:, n_in:], skip_special_tokens=True)

def predict_batch(model, tokenizer, formulas):
    results = []
    for i in range(0, len(formulas), BATCH_SIZE):
        chunk = formulas[i:i + BATCH_SIZE]
        texts = [build_prompt(tokenizer, f) for f in chunk]
        t0 = time.time()
        try:
            responses = generate_texts(model, tokenizer, texts)
        except torch.cuda.OutOfMemoryError:   # auto-fallback to serial
            torch.cuda.empty_cache()
            print("  [warn] OOM - falling back to batch size 1 for this chunk")
            responses = [generate_texts(model, tokenizer, [t])[0] for t in texts]
        dt = time.time() - t0
        for f, resp in zip(chunk, responses):
            data, clean = parse_response(resp)
            results.append((f, data, clean))
        print(f"  batch {i // BATCH_SIZE + 1}: {len(chunk)} alloys in {dt:.1f}s "
              f"({len(chunk) / dt:.2f} samples/s)")
    return results

def apply_ood_guardrails(formula, data):
    flags = []
    ti = re.search(r'Ti([0-9\.]+)', formula)
    ti_at = float(ti.group(1)) if ti else 0.0
    if ti_at > 50.0:
        flags.append("RHEA_EXTRAPOLATION")     # trained on W-Mo-Nb-Ta RHEAs, not Ti
    v = re.search(r'V([0-9\.]+)', formula); mo = re.search(r'Mo([0-9\.]+)', formula)
    mo_eq = (float(mo.group(1)) if mo else 0.0) + (float(v.group(1)) if v else 0.0) / 1.5
    phase = str(data.get("phase", "")).lower()
    # FIXED logic: 'beta' now included in the inner check
    if mo_eq < 8.0 and any(k in phase for k in ("bcc", "beta", "single")):
        flags.append("PHASE_HALLUCINATION")    # lean-stabilizer alloy cannot be single-phase beta
    data["Mo_eq"] = round(mo_eq, 2)
    data["OOD_FLAGS"] = flags if flags else ["NONE"]
    return data

if __name__ == "__main__":
    model, tokenizer = load_model()

    # --- single-alloy demo ---
    print("\n--- SINGLE: Ti90Al6V4 ---")
    for f, data, clean in predict_batch(model, tokenizer, ["Ti90Al6V4"]):
        print("raw answer:", clean)
        print("guarded:", json.dumps(apply_ood_guardrails(f, data), indent=2))

    # --- batch demo on top CrabNet candidates ---
    print("\n--- BATCH: top CrabNet shortlist ---")
    batch = ["Ti85V15", "Ti90V9Fe1", "Ti90V9Cr1", "Ti90Al7V3", "Ti81Al9Mo2V5Cr3", "Ti84V9Fe2Al5"]
    for f, data, clean in predict_batch(model, tokenizer, batch):
        print(f"{f:>16}: {json.dumps(apply_ood_guardrails(f, data))}")
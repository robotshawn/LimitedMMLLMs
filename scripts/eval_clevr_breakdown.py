import json
from collections import defaultdict
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

BASE_ID = "llava-hf/llava-1.5-7b-hf"
LORA_DIR = "out/llava_clevr_qlora"
VAL_JSONL = "data/processed/clevr_val_2k.jsonl"

YESNO = {"yes","no"}
COLORS = {"gray","red","blue","green","brown","purple","cyan","yellow"}
SHAPES = {"cube","sphere","cylinder"}
SIZES  = {"small","large"}
MATS   = {"rubber","metal"}

def norm(s: str) -> str:
    return s.strip().lower().replace(".", "")

def bucket(gt: str) -> str:
    g = norm(gt)
    if g in YESNO:
        return "yesno"
    if g.isdigit():
        return "count"
    if g in COLORS: return "color"
    if g in SHAPES: return "shape"
    if g in SIZES:  return "size"
    if g in MATS:   return "material"
    return "other"

@torch.no_grad()
def main():
    processor = AutoProcessor.from_pretrained(LORA_DIR)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = LlavaForConditionalGeneration.from_pretrained(
        BASE_ID,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, LORA_DIR).eval()
    device = next(model.parameters()).device

    stat = defaultdict(lambda: {"n":0, "c":0})

    with open(VAL_JSONL, "r") as f:
        for line in f:
            ex = json.loads(line)
            img = Image.open(ex["image"]).convert("RGB")
            q = ex["messages"][0]["content"][1]["text"]
            gt = ex["messages"][1]["content"][0]["text"]

            prompt = f"USER: <image>\n{q}\nASSISTANT:"
            inputs = processor(text=prompt, images=img, return_tensors="pt")
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
            txt = processor.batch_decode(out, skip_special_tokens=True)[0]
            pred = txt.split("ASSISTANT:")[-1].strip()

            b = bucket(gt)
            stat[b]["n"] += 1
            stat[b]["c"] += int(norm(pred) == norm(gt))

    print("=== Breakdown (exact match) ===")
    total_n = sum(v["n"] for v in stat.values())
    total_c = sum(v["c"] for v in stat.values())
    for k in sorted(stat.keys()):
        n, c = stat[k]["n"], stat[k]["c"]
        print(f"{k:10s}  {c:4d}/{n:4d}  acc={c/n:.4f}")
    print(f"ALL        {total_c:4d}/{total_n:4d}  acc={total_c/total_n:.4f}")

if __name__ == "__main__":
    main()

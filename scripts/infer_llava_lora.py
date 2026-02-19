import json
import random
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

BASE_ID = "llava-hf/llava-1.5-7b-hf"
LORA_DIR = "out/llava_clevr_qlora"
VAL_JSONL = "data/processed/clevr_val_2k.jsonl"

def normalize_ans(s: str) -> str:
    return s.strip().lower().replace(".", "")

@torch.no_grad()
def main():
    processor = AutoProcessor.from_pretrained(LORA_DIR)

    # ✅ 关键：推理也用 4bit，让 base 完整落到 GPU，避免 CPU offload + accelerate balancing
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
        device_map={"": 0},   # 强制单卡
    )

    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.eval()
    device = next(model.parameters()).device

    N = 20
    lines = open(VAL_JSONL, "r").read().splitlines()
    picks = random.sample(lines, k=min(N, len(lines)))

    hit = 0
    for idx, line in enumerate(picks, 1):
        ex = json.loads(line)
        img = Image.open(ex["image"]).convert("RGB")
        q = ex["messages"][0]["content"][1]["text"]
        gt = ex["messages"][1]["content"][0]["text"]

        prompt = f"USER: <image>\n{q}\nASSISTANT:"
        inputs = processor(text=prompt, images=img, return_tensors="pt")
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        out = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )
        txt = processor.batch_decode(out, skip_special_tokens=True)[0]
        pred = txt.split("ASSISTANT:")[-1].strip()

        ok = normalize_ans(pred) == normalize_ans(gt)
        hit += int(ok)

        print("=" * 80)
        print(f"[{idx}] Q : {q}")
        print(f"     GT: {gt}")
        print(f"     PR: {pred}")
        print(f"     OK: {ok}")

    print("=" * 80)
    print(f"Sample accuracy: {hit}/{len(picks)} = {hit/len(picks):.3f}")

if __name__ == "__main__":
    main()

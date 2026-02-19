import json
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

BASE_ID = "llava-hf/llava-1.5-7b-hf"
LORA_DIR = "out/llava_clevr_qlora"
VAL_JSONL = "data/processed/clevr_val_2k.jsonl"

def norm(s: str) -> str:
    return s.strip().lower().replace(".", "")

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
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.eval()
    device = next(model.parameters()).device

    total = 0
    correct = 0

    with open(VAL_JSONL, "r") as f:
        for line in f:
            ex = json.loads(line)
            img = Image.open(ex["image"]).convert("RGB")
            q = ex["messages"][0]["content"][1]["text"]
            gt = ex["messages"][1]["content"][0]["text"]

            prompt = f"USER: <image>\n{q}\nASSISTANT:"
            inputs = processor(text=prompt, images=img, return_tensors="pt")
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            out = model.generate(
                **inputs,
                max_new_tokens=8,   # CLEVR 答案很短，越小越快
                do_sample=False,
            )
            txt = processor.batch_decode(out, skip_special_tokens=True)[0]
            pred = txt.split("ASSISTANT:")[-1].strip()

            total += 1
            correct += int(norm(pred) == norm(gt))

            if total % 200 == 0:
                print(f"{total}  acc={correct/total:.3f}")

    print(f"Final: {correct}/{total}  acc={correct/total:.4f}")

if __name__ == "__main__":
    main()

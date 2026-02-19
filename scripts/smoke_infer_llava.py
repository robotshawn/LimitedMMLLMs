import json
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

def main():
    # 读一条样本
    with open("data/processed/clevr_val_2k.jsonl", "r") as f:
        ex = json.loads(next(f))

    img = Image.open(ex["image"]).convert("RGB")
    question = ex["messages"][0]["content"][1]["text"]

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # 按模型卡推荐模板 + <image> 更稳（避免不同版本的 chat template 差异）:contentReference[oaicite:5]{index=5}
    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=32)
    txt = processor.batch_decode(out, skip_special_tokens=True)[0]
    print("Q:", question)
    print("GT:", ex["messages"][1]["content"][0]["text"])
    print("Pred:", txt)

if __name__ == "__main__":
    main()


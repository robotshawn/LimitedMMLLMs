import inspect
from typing import List, Dict, Any

import torch
from PIL import Image
from datasets import load_dataset

from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)

MODEL_ID = "llava-hf/llava-1.5-7b-hf"


def build_collator(processor, max_length: int = 1024):
    """
    你的 JSONL 样本格式：
      ex["image"] : 图片绝对路径
      ex["messages"][0] : user (content: [image, text])
      ex["messages"][1] : assistant (content: [text])

    训练模板（LLaVA 常用）：
      USER: <image>\n{question}\nASSISTANT: {answer}

    labels 只对 answer 计算 loss：prompt/pad 全 mask=-100
    """
    tok = processor.tokenizer

    def collate(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        imgs, questions, answers = [], [], []

        for ex in examples:
            imgs.append(Image.open(ex["image"]).convert("RGB"))
            questions.append(ex["messages"][0]["content"][1]["text"])
            answers.append(ex["messages"][1]["content"][0]["text"])

        prompts = [f"USER: <image>\n{q}\nASSISTANT:" for q in questions]
        full_texts = [p + " " + a for p, a in zip(prompts, answers)]

        inputs = processor(
            text=full_texts,
            images=imgs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        labels = inputs["input_ids"].clone()

        pad_id = tok.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # mask prompt
        for i, p in enumerate(prompts):
            p_ids = tok(p, add_special_tokens=True).input_ids
            labels[i, : len(p_ids)] = -100

        inputs["labels"] = labels
        return inputs

    return collate


def _filter_kwargs_for_init(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """只保留 cls.__init__ 支持的参数，避免 unexpected keyword"""
    sig = set(inspect.signature(cls.__init__).parameters.keys())
    return {k: v for k, v in kwargs.items() if k in sig}


def make_training_args(out_dir: str, eval_steps: int = 200, save_steps: int = 200) -> TrainingArguments:
    """
    兼容不同 transformers 版本：
      - evaluation_strategy vs eval_strategy
      - 可能存在 set_evaluate / set_save
    """
    base = dict(
        output_dir=out_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        num_train_epochs=1,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=20,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    ctor_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    if "eval_strategy" in ctor_params:
        base["eval_strategy"] = "steps"
    elif "evaluation_strategy" in ctor_params:
        base["evaluation_strategy"] = "steps"

    if "save_strategy" in ctor_params:
        base["save_strategy"] = "steps"

    args = TrainingArguments(**_filter_kwargs_for_init(TrainingArguments, base))

    # 新版链式 API 兜底
    if not (("eval_strategy" in ctor_params) or ("evaluation_strategy" in ctor_params)):
        if hasattr(args, "set_evaluate"):
            args = args.set_evaluate(strategy="steps", steps=eval_steps)

    if "save_strategy" not in ctor_params:
        if hasattr(args, "set_save"):
            args = args.set_save(strategy="steps", steps=save_steps)

    return args


def main():
    # 0) 数据
    train_path = "data/processed/clevr_train_20k.jsonl"
    val_path = "data/processed/clevr_val_2k.jsonl"
    ds = load_dataset("json", data_files={"train": train_path, "eval": val_path})

    # 1) processor
    processor = AutoProcessor.from_pretrained(MODEL_ID)  # 如需慢处理器：use_fast=False

    # 2) QLoRA 4bit
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # 3) LoRA
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    # 4) Trainer
    out_dir = "out/llava_clevr_qlora"
    args = make_training_args(out_dir=out_dir, eval_steps=200, save_steps=200)

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        data_collator=build_collator(processor, max_length=1024),
    )

    # 兼容不同 transformers Trainer 版本：tokenizer / processing_class 可能存在或不存在
    trainer_params = set(inspect.signature(Trainer.__init__).parameters.keys())
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = processor.tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = processor

    trainer = Trainer(**_filter_kwargs_for_init(Trainer, trainer_kwargs))

    # 5) 训练 + 保存
    trainer.train()
    model.save_pretrained(out_dir)       # 保存 LoRA adapter
    processor.save_pretrained(out_dir)   # 保存 processor
    print(f"[OK] Saved to: {out_dir}")


if __name__ == "__main__":
    main()

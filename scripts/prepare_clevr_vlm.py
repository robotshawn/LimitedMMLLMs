import argparse, json, os, random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clevr_root", type=str, required=True, help=".../CLEVR_v1.0")
    ap.add_argument("--split", type=str, choices=["train", "val"], required=True)
    ap.add_argument("--out", type=str, required=True, help="output jsonl")
    ap.add_argument("--max_samples", type=int, default=0, help="0 means all")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    clevr_root = Path(args.clevr_root)

    q_path = clevr_root / "questions" / f"CLEVR_{args.split}_questions.json"
    img_dir = clevr_root / "images" / args.split

    with open(q_path, "r") as f:
        questions = json.load(f)["questions"]

    # 受限场景下更“稳”的做法：先抽一部分跑通；之后再放大
    if args.max_samples and args.max_samples > 0:
        questions = random.sample(questions, k=min(args.max_samples, len(questions)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_missing = 0
    with open(out_path, "w") as w:
        for q in questions:
            img_file = q["image_filename"]
            img_path = img_dir / img_file
            if not img_path.exists():
                n_missing += 1
                continue

            sample = {
                "image": str(img_path.resolve()),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": q["question"]},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": q["answer"]},
                        ],
                    },
                ],
                # 可选：把 program / question_family_index 留着，后续做子任务评估很方便
                "meta": {
                    "question_index": q.get("question_index"),
                    "question_family_index": q.get("question_family_index"),
                    "program": q.get("program"),
                },
            }
            w.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[OK] wrote: {out_path}  samples={len(questions)-n_missing}  missing_images={n_missing}")

if __name__ == "__main__":
    main()


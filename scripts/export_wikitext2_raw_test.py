from datasets import load_dataset
from pathlib import Path
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HF wikitext-2-raw-v1 split as llama.cpp raw text")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"], help="dataset split")
    parser.add_argument("-o", "--output", required=True, help="output raw text file")
    args = parser.parse_args()

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=args.split)
    text = "\n\n".join(ds["text"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    print(f"Saved {args.split} split to {out}")
    print(f"Characters: {len(text)}")
    print(f"Rows: {len(ds)}")


if __name__ == "__main__":
    main()

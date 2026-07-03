"""
prepare_data.py
----------------
Parses Chat_Data_for_assessment_of_applicants.json (a concatenation of
JSON objects that is NOT valid JSONL — some entries are pretty-printed
across multiple lines, some have trailing commas, some have an "id"/"tags"
wrapper and some don't) and turns it into clean, trainable JSONL splits.

Usage:
    python prepare_data.py --input Chat_Data_for_assessment_of_applicants.json \
                            --out_dir data/ --val_ratio 0.1
"""
import json
import argparse
import random
from pathlib import Path


def parse_concatenated_json(text: str):
    """Stream-decode a file that contains multiple back-to-back JSON
    objects (optionally separated by commas/whitespace/newlines)."""
    decoder = json.JSONDecoder()
    idx, n = 0, len(text)
    objs = []
    while idx < n:
        while idx < n and text[idx] in " \n\r\t,":
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(text, idx)
        objs.append(obj)
        idx = end
    return objs


def validate(obj):
    """Basic schema check: must have a non-empty 'messages' list with
    alternating roles starting with system/user."""
    msgs = obj.get("messages")
    if not msgs or not isinstance(msgs, list):
        return False
    roles = [m.get("role") for m in msgs]
    if roles[0] != "system":
        return False
    if any(not m.get("content", "").strip() for m in msgs):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    objs = parse_concatenated_json(text)

    clean = [o for o in objs if validate(o)]
    dropped = len(objs) - len(clean)
    print(f"Parsed {len(objs)} objects, {len(clean)} valid, {dropped} dropped.")

    random.seed(args.seed)
    random.shuffle(clean)
    n_val = max(1, int(len(clean) * args.val_ratio))
    val, train = clean[:n_val], clean[n_val:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for o in train:
            f.write(json.dumps({"messages": o["messages"]}, ensure_ascii=False) + "\n")
    with open(out_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for o in val:
            f.write(json.dumps({"messages": o["messages"]}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train)} train examples -> {out_dir/'train.jsonl'}")
    print(f"Wrote {len(val)} val examples   -> {out_dir/'val.jsonl'}")


if __name__ == "__main__":
    main()

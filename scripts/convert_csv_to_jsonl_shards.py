#!/usr/bin/env python3
"""Convert large CSV into JSONL shard files suitable for Hugging Face `datasets` streaming.

Creates numbered shards in the output directory.
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def row_to_example(row, prompt_template=None, target_field="expected_throughput_mbps"):
    prompt = prompt_template.format(**row) if prompt_template else json.dumps(row)
    target = row.get(target_field)
    return {"prompt": prompt, "target": target}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", default="data/jsonl_shards")
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--prompt-template", default=None)
    parser.add_argument("--target-field", default="expected_throughput_mbps")
    args = parser.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    reader = pd.read_csv(args.csv, chunksize=args.shard_size)
    i = 0
    for chunk_idx, chunk in enumerate(reader):
        shard_path = outdir / f"shard-{chunk_idx:04d}.jsonl"
        with shard_path.open("w", encoding="utf-8") as fh:
            for _, row in chunk.iterrows():
                ex = row_to_example(row.to_dict(), prompt_template=args.prompt_template, target_field=args.target_field)
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {shard_path}")
        i += 1

    print(f"Wrote {i} shards to {outdir}")


if __name__ == '__main__':
    main()

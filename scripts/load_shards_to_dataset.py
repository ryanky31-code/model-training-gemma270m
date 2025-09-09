#!/usr/bin/env python3
"""Load JSONL shards into a Hugging Face `datasets` Dataset in streaming mode.

This script demonstrates how to load the `data/jsonl_shards/*.jsonl` produced earlier.
"""
import argparse
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False


def print_sample_from_jsonl(pattern):
    import glob, json
    files = sorted(glob.glob(pattern))
    for f in files[:3]:
        with open(f, 'r', encoding='utf-8') as fh:
            for i, line in enumerate(fh):
                if not line.strip():
                    continue
                obj = json.loads(line)
                print(f"{f}:{i}", obj)
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shards-pattern', default='data/jsonl_shards/*.jsonl')
    parser.add_argument('--stream', action='store_true')
    args = parser.parse_args()

    if HAS_DATASETS:
        ds = load_dataset('json', data_files=args.shards_pattern, streaming=args.stream)
        print(ds)
        it = ds['train'] if 'train' in ds else ds
        for i, ex in enumerate(it):
            print(i, ex)
            if i >= 2:
                break
    else:
        print("datasets package not available; falling back to simple JSONL preview")
        print_sample_from_jsonl(args.shards_pattern)


if __name__ == '__main__':
    main()

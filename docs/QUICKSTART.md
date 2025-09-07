Quickstart Cheat-sheet

This quick cheat-sheet summarizes the minimal commands to get started (smoke tests). Keep it handy.

1. Create venv and install smoke deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Generate a small smoke CSV

```bash
python scripts/generate_synthetic_smoke.py
```

3. Validate dataset (optional)

```bash
python scripts/validate_dataset.py synthetic_wifi_5ghz_outdoor_smoke.csv
```

4. Dry-run the trainer to validate pipeline

```bash
python scripts/finetune_gemma_from_csv.py --csv synthetic_wifi_5ghz_outdoor_smoke.csv --dry-run --max-rows 50
```

5. Run a LoRA smoke training

```bash
python scripts/finetune_gemma_from_csv.py --csv synthetic_wifi_5ghz_outdoor_smoke.csv --mode lora --lora-r 8 --lora-alpha 32 --num-epochs 1 --per-device-batch-size 2 --max-rows 200
```

6. Convert CSV to JSONL shards (for streaming)

```bash
python scripts/convert_csv_to_jsonl_shards.py --csv data/synthetic_wifi_5ghz_10,000.csv --out-dir data/jsonl_shards --shard-size 2500
```

7. Run tests

```bash
pytest -q -r a
```

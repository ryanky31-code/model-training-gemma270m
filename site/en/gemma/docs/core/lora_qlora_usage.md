LoRA / QLoRA usage examples

The repository includes `scripts/finetune_gemma_from_csv.py` with `--mode` choices: `full`, `lora`, `qlora` and additional LoRA hyperparameters.

Example: LoRA quick smoke (uses adapters, small epochs)

```bash
python scripts/finetune_gemma_from_csv.py \
  --csv data/synthetic_wifi_5ghz_10,000.csv \
  --base-model google/gemma-3-270m-it \
  --checkpoint-dir ./checkpoints/lora-smoke \
  --mode lora \
  --lora-r 8 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --num-epochs 1 \
  --per-device-batch-size 2 \
  --max-rows 200
```

Example: QLoRA (Colab recommended)

```bash
# In Colab: install bitsandbytes & peft, then run with mode qlora
%pip install bitsandbytes peft
python scripts/finetune_gemma_from_csv.py \
  --csv /content/synthetic_wifi_5ghz_outdoor.csv \
  --base-model google/gemma-3-270m-it \
  --checkpoint-dir /content/checkpoints/qlora \
  --mode qlora \
  --num-epochs 2 \
  --per-device-batch-size 4
```

Notes:
- QLoRA requires `bitsandbytes` and `peft`. The script will warn and fall back to full fine-tune if those aren't installed in the environment.
- For real experiments use larger `--num-epochs` and remove `--max-rows`.

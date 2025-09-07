# model-training-gemma270m

Detailed README — Gemma-3 (270M) fine-tuning from synthetic CSV data

This repository demonstrates an end-to-end workflow for generating a synthetic wireless dataset, converting it into a conversational format, and fine-tuning Google's Gemma-3 270M model using Hugging Face Transformers + TRL (SFT). The repository includes both notebook and script-based approaches so you can run quick smoke tests locally or full experiments in Colab/remote GPU environments.

Contents
- `site/en/gemma/docs/core/huggingface_text_full_finetune.ipynb` — upstream Colab example from Google (untouched in this repo). Use as reference for the TRL-based SFT flow.
- `site/en/gemma/docs/core/huggingface_text_full_finetune_with_generator.ipynb` — Colab-ready notebook that embeds the synthetic dataset generator and adapts the example to train on the generated CSV.
- `scripts/generate_synthetic_smoke.py` — tiny smoke generator that produces a small CSV and ZIP (quick test).
- `scripts/finetune_gemma_from_csv.py` — command-line script that converts a CSV into a conversational dataset and runs TRL's `SFTTrainer` (full-training script).
- `scripts/fix_notebook_widgets.py` — small utility used to strip problematic `metadata.widgets` entries so GitHub can render notebooks correctly.
- `requirements.txt` — minimal Python dependencies required for the training pipeline.

Quick start (local smoke test)
1. Create and activate a Python virtual environment (recommended):

   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies (for smoke tests you only need the data libs; full training requires `torch`, `transformers`, `trl` etc.):

   pip install -r requirements.txt

3. Run the tiny smoke generator (creates `./synthetic_wifi_5ghz_outdoor_smoke.csv`):

   python scripts/generate_synthetic_smoke.py

4. Inspect the CSV and run the CSV→dataset conversion using the CLI script (no heavy model downloads required for this step):

   python scripts/finetune_gemma_from_csv.py --csv synthetic_wifi_5ghz_outdoor_smoke.csv --max-rows 50

   Note: `--max-rows` prevents full training; the script defers heavy imports until runtime so you can validate argument parsing without installing all heavy libs.

Running in Google Colab (recommended for GPU training)
1. Open the notebook in Colab:

   https://colab.research.google.com/github/ryanky31-code/model-training-gemma270m/blob/main/site/en/gemma/docs/core/huggingface_text_full_finetune_with_generator.ipynb

2. Change Runtime → Change runtime type → Hardware accelerator → GPU. Prefer an L4/A100/T4 for performance (L4/A100 recommended for bf16/flash-attn support).

3. Install dependencies in the first cell (uncomment and run):

   %pip install torch transformers datasets trl accelerate tensorboard sentencepiece protobuf pandas numpy

   Optional (flash attention on supported GPUs):

   %pip install flash-attn

4. Accept the Gemma model license on Hugging Face:

   Visit https://huggingface.co/google/gemma-3-270m-it and press "Agree and access repository".

5. Login to Hugging Face in Colab (use input prompt or Colab secrets):

   from huggingface_hub import login
   login()

6. (Optional) Mount Google Drive to persist checkpoints:

   from google.colab import drive
   drive.mount('/content/drive')

7. Run the notebook cells in order:
   - Install dependencies
   - Generate dataset (adjust `N_SAMPLES` to control size)
   - Convert CSV → dataset
   - Load model & tokenizer (requires HF token / license acceptance)
   - Configure `SFTConfig` and build `SFTTrainer`
   - Run `trainer.train()` and `trainer.save_model()`
   - Run inference/test cells to inspect outputs

Design notes and how the pieces fit
- Dataset generator: produces synthetic P2P wireless link scenarios with environmental/weather/obstruction parameters. Output fields include `recommended_channel_mhz` and `expected_throughput_mbps` that are used as training targets.
- CSV → conversational dataset transformer: converts each CSV row into a two-message conversation: a `user` message describing the scenario and a `assistant` message containing the target (e.g., recommended channel). This format matches the Gemma chat template used by the original example.
- Training: Uses TRL's `SFTTrainer` to perform supervised fine-tuning with the same API used in the official Gemma example. The default script performs full model fine-tuning (not LoRA) — for resource-constrained runs prefer LoRA/QLoRA.

Smoke vs real experiments
- Smoke tests: use `N_SAMPLES` small (50–200) and `num_train_epochs=1`; validate end-to-end pipeline without heavy cost.
- Real experiments: use a larger `N_SAMPLES` (10k+), tune `num_train_epochs`, batch sizes, and learning rate. Persist checkpoints to Drive and monitor logs in TensorBoard.

Common issues & troubleshooting
- Notebook render errors on GitHub: some notebook metadata (widgets) can cause GitHub to fail rendering. Use `scripts/fix_notebook_widgets.py <notebook.ipynb>` to remove those entries.
- HF model download errors: ensure you accepted the license on the model page and your HF token has the necessary access. Use `huggingface_hub.login()` to provide the token.
- OOM during model load or training: reduce `per_device_train_batch_size`, reduce `max_length`, enable `gradient_checkpointing`, or use bf16/fp16 if your GPU supports it. For severe limits, switch to LoRA/QLoRA.
- TRL / fused optimizers: if `adamw_torch_fused` or other fused ops fail, switch to standard `adamw_torch` or install matching PyTorch builds.

Next steps and suggestions
- Add explicit automated tests for CSV→dataset conversion. I can add a unit test that validates the first few records are converted to the expected message format.
- Add a Colab cell to optionally push prepared dataset and final model to the Hugging Face hub (requires `push_to_hub=True` and token with write access).
- Add a short example that shows how to switch to LoRA/QLoRA to reduce GPU memory usage.

Contact / support
If you hit errors running in Colab or locally, paste the failing cell output here and I will suggest exact fixes (dependency pins, memory tweaks, or code changes).

License
This repo contains example code adapted from Google Colab examples; follow the original model and code licenses (Apache 2.0 where indicated) and respect Hugging Face model licensing for Gemma.

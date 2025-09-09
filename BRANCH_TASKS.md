Branch: dev/colab-workflow — Ordered Task Plan (dependencies-first)

Goal: implement the Colab/data/experiment workflow in dependency order. Each Main Task lists ordered subtasks. Start with foundational items that other tasks depend on.

Legend: [x] done — [ ] pending
Execution note: tasks below are tagged with [Agent] if they can be implemented and executed by the agent in this repo/devcontainer, or [Colab/Human] if they require interactive Colab (GPU, Drive mount, HF token in Colab) or user actions.

Main Task 1 — Project & environment foundations (all other work depends on this)
- [x] 1.1 Verify branch and working tree
  - git checkout dev/colab-workflow
  - Confirm working tree is clean and up to date
- [x] 1.2 Reproducible environment
  - [x] Ensure `requirements.txt` lists the minimal pinned dependencies (present)
  - [x] Add `environment.md` with exact Colab and local setup commands and known-good package versions (`environment.md` added)
- [x] 1.3 Notebook hygiene and git safety
  - [x] Add `scripts/fix_notebook_widgets.py` (present and improved)
  - [x] Run the fixer over notebooks to ensure GitHub rendering (done)
  - [x] Install pre-commit hook that checks/normalizes notebooks (configured to fail when changes are required)
- [x] 1.4 Git hooks installer and repo hooksPath configured
  - Added `scripts/install_git_hooks.sh` and set `core.hooksPath` to `.githooks` (installed)
  - [Agent]

Main Task 2 — Data generation core (depends on Task 1)
- [x] 2.1 Large-scale batched generator (authoritative data source)
  - Implement `scripts/generate_synthetic_large.py` with params: `--n-samples`, `--chunk-size`, `--seed`, `--out-dir` (implemented)
  - Write CSV in streaming chunks and compute SHA256 per chunk/file (done)
  - [Agent]
- [x] 2.2 Stratified sampling & balancing utilities
  - Add `--stratify-by` option and simple oversample/undersample modes to control class balance (implemented)
  - [Agent]
- [x] 2.3 Storage & integrity
  - Save final CSV + ZIP atomically and produce SHA256 for verification (done)
  - (Optional) Add `--upload-to-drive` or `--upload-to-hf` flags (deferred)
  - [Agent]
- [x] 2.4 Keep `scripts/generate_synthetic_smoke.py` for quick smoke tests (present and executed)
  - [Agent]
- [x] 2.5 Generated large dataset artifact (10k CSV)
  - Created `data/synthetic_wifi_5ghz_10,000.csv` and zip with SHA256 (done)
  - [Agent]

Main Task 3 — Data validation & conversion (depends on Task 2)
- [x] 3.1 Validation utilities
  - Add `scripts/validate_dataset.py` to check for missing targets, NaNs, range checks, and value distributions (implemented)
  - [Agent]
- [x] 3.2 CSV → HF conversational Dataset converter
  - Implement streaming/batched conversion to a Hugging Face `Dataset` to avoid large memory usage (converter implemented as JSONL shards)
  - Add CLI options for `--target-field` and `--prompt-template` (CLI provided)
  - [Agent]
- [x] 3.3 Unit tests for conversion
  - Add a unit test in `tests/test_conversion.py` that checks prompt and target formatting (implemented and passing)
  - [Agent]
- [ ] 3.4 Install `datasets` in devcontainer (optional)
  - Add instructions or devcontainer setup to install `datasets` so streaming ingestion can be tested locally
  - [Agent]

 - [x] 3.4 Install `datasets` in devcontainer (documentation)
  - Added `requirements-dev.txt` with `datasets` and documented installation in `environment.md` (done)
  - [Agent]
 - [x] 3.5 JSONL shard conversion
  - Implemented `scripts/convert_csv_to_jsonl_shards.py` and produced JSONL shards for the 10k CSV (done)
  - [Agent]

Main Task 4 — Baselines & heuristics (depends on Task 3)
- [ ] 4.1 Implement baseline predictors
  - Random baseline, frequency baseline, mean-throughput baseline
- [ ] 4.2 Baseline evaluation harness
  - Produce CSV/markdown summary of baseline metrics for quick comparisons

 - [x] 4.1 Implement baseline predictors
   - Random baseline, frequency baseline, mean-throughput baseline (implemented in `scripts/baselines.py`)
 - [x] 4.2 Baseline evaluation harness
   - Produce CSV/markdown summary of baseline metrics for quick comparisons (CLI supports CSV/JSON outputs)

Main Task 5 — Training infrastructure (depends on Task 3 + Task 1)
- [x] 5.1 Training CLI scaffolding (basic CLI present: `scripts/finetune_gemma_from_csv.py`)
  - Extend `scripts/finetune_gemma_from_csv.py` with modes: `--mode full|lora|qlora`, `--max-rows`, `--out-dir`, `--resume` (extend as needed)
  - [Agent]
- [ ] 5.2 LoRA / QLoRA support [IN-PROGRESS: Agent]
  - Add notebook cells and script code paths to enable LoRA and QLoRA training with recommended defaults (scaffolding added)
  - [Agent implement + Colab run required]
 - [x] 5.2 LoRA / QLoRA support
  - Notebook cells and script scaffolding added; Colab run still required for full QLoRA execution (Agent scaffolding done)
  - [Agent implement + Colab run required]
 - [x] 5.3 Resource & safety flags
  - Added `--fp16/--bf16`, `--gradient-checkpointing`, and related flags to CLI (done)
  - [Agent]
 - [x] 5.4 Checkpointing & persistence
  - Added `--resume-from-checkpoint`, `--save-strategy`, `--save-steps`, and `--save-total-limit` to CLI (done)
  - [Agent]
 - [x] 5.5 Add `--dry-run` mode to training CLI
  - Implemented `--dry-run` which prepares datasets and skips heavy imports/trainer construction (done)
  - [Agent]

Main Task 6 — Training experiments & orchestration (depends on Task 5)
- [ ] 6.1 Learning-curve orchestration
  - `scripts/run_learning_curve.py` runs training over increasing dataset sizes and records metrics
- [ ] 6.2 Simple hyperparameter sweep helper
  - Add a small wrapper to launch a few hyperparameter combinations and save results
- [ ] 6.3 Smoke train / env checks
  - `--dry-run` mode to build trainer and verify no immediate failures (but skip heavy `.train()`) (pending: tie to 5.5)

Main Task 7 — Evaluation & analysis (depends on Task 6 + Task 4)
- [ ] 7.1 Official evaluation script
  - `scripts/evaluate_model.py` — exact-match, top-k, confusion matrix, MAE/RMSE, per-class metrics
- [ ] 7.2 Visualization & reporting
  - Generate learning-curve plots, confusion matrix heatmaps, and a markdown summary per experiment
- [ ] 7.3 Baseline vs model comparison
  - Automate side-by-side comparison with baselines and produce final summary table

Main Task 8 — Colab polish & one-click demo (depends on Task 5 + Task 7)
- [ ] 8.1 Notebook parameter toggles
  - Add a top-of-notebook config cell for mode, N_SAMPLES, N_EPOCHS, BATCH_SIZE, OUT_DIR
- [ ] 8.2 One-click demo cell
  - A cell that runs generator → conversion → quick train → evaluate (with small N_SAMPLES)
- [ ] 8.3 Embed LoRA/QLoRA usage into the Colab notebook
  - `site/en/gemma/docs/core/lora_qlora_usage.md` added as a usage note; embed as a code cell in the notebook for convenience (pending)
 - [x] 8.3 Embed LoRA/QLoRA usage into the Colab notebook
  - Embedded LoRA/QLoRA usage as a markdown cell in `huggingface_text_full_finetune_with_generator.ipynb` (done)
  - [Agent]
 - [x] 8.4 Drive mount and HF token handling (user accepted HF license and added token to Colab keys — no further action needed)

Main Task 9 — Tests, CI and merge prep (depends on finished code)
 - [x] 9.1 Unit tests
  - Add tests for generator determinism, conversion output, and validator (conversion tests, validator, and generator determinism tests added and passing)
- [ ] 9.2 Lightweight CI
  - Configure CI to run unit tests and smoke data generation only; avoid heavy GPU tasks
 - [x] 9.2 Lightweight CI
  - Added `.github/workflows/ci.yml` to run `pytest` and a smoke generator job (done)
  - [Agent]
 - [ ] 9.3 Merge checklist
  - Ensure notebooks render, tests pass, README updated, artifacts documented
- [ ] 9.4 Add GitHub Actions workflow
  - Create a lightweight GitHub Actions workflow that runs `pytest`, the notebook fixer, and the smoke generator (small sample) on push/PR
- [ ] 9.5 Add CI `--dry-run` job
  - Add a job that uses the training CLI with `--max-rows` and `--dry-run` to validate trainer construction (no heavy training)
 - [x] 9.4 Add GitHub Actions workflow
  - Workflow file created at `.github/workflows/ci.yml` (done)
 - [x] 9.5 Add CI `--dry-run` job
  - Added `dry-run-trainer` job that runs the training CLI in `--dry-run` mode (done)
 - [x] 9.6 Add PR template with CI badge
  - Added `.github/PULL_REQUEST_TEMPLATE.md` including the CI badge (done)

Main Task 10 — Documentation & optional push (least dependent)
- [x] 10.1 Keep `BRANCH_TASKS.md` and `README.md` updated (BRANCH_TASKS.md edited; README previously updated)
- [ ] 10.2 Add examples/how-to docs (LoRA vs full fine-tune, evaluation guide)
- [ ] 10.3 Add `push_to_hub` option only after final validation and license checks

Usage notes
- Implement tasks in order; starting points: 1.2 (environment.md) and 2.1 (generate_synthetic_large.py).
- Completed so far: environment.md, notebook fixer, pre-commit hook (configured), small smoke generator and smoke run, branch created and changes pushed, HF token present in Colab.

I can start implementing 2.1 (`scripts/generate_synthetic_large.py`) next unless you'd prefer a different task.


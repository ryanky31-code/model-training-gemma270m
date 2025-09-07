Branch: dev/colab-workflow — Ordered Task Plan (dependencies-first)

Goal: implement the Colab/data/experiment workflow in dependency order. Each Main Task lists ordered subtasks. Start with foundational items that other tasks depend on.

Legend: [x] done — [ ] pending

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

Main Task 2 — Data generation core (depends on Task 1)
- [x] 2.1 Large-scale batched generator (authoritative data source)
  - Implement `scripts/generate_synthetic_large.py` with params: `--n-samples`, `--chunk-size`, `--seed`, `--out-dir` (implemented)
  - Write CSV in streaming chunks and compute SHA256 per chunk/file (done)
- [ ] 2.2 Stratified sampling & balancing utilities
  - Add `--stratify-by` option and simple oversample/undersample modes to control class balance
- [x] 2.3 Storage & integrity
  - Save final CSV + ZIP atomically and produce SHA256 for verification (done)
  - (Optional) Add `--upload-to-drive` or `--upload-to-hf` flags (deferred)
- [x] 2.4 Keep `scripts/generate_synthetic_smoke.py` for quick smoke tests (present and executed)

Main Task 3 — Data validation & conversion (depends on Task 2)
- [x] 3.1 Validation utilities
  - Add `scripts/validate_dataset.py` to check for missing targets, NaNs, range checks, and value distributions (implemented)
- [x] 3.2 CSV → HF conversational Dataset converter
  - Implement streaming/batched conversion to a Hugging Face `Dataset` to avoid large memory usage (converter implemented as JSONL shards)
  - Add CLI options for `--target-field` and `--prompt-template` (CLI provided)
- [x] 3.3 Unit tests for conversion
  - Add a unit test in `tests/test_conversion.py` that checks prompt and target formatting (implemented and passing)

Main Task 4 — Baselines & heuristics (depends on Task 3)
- [ ] 4.1 Implement baseline predictors
  - Random baseline, frequency baseline, mean-throughput baseline
- [ ] 4.2 Baseline evaluation harness
  - Produce CSV/markdown summary of baseline metrics for quick comparisons

Main Task 5 — Training infrastructure (depends on Task 3 + Task 1)
- [x] 5.1 Training CLI scaffolding (basic CLI present: `scripts/finetune_gemma_from_csv.py`)
  - Extend `scripts/finetune_gemma_from_csv.py` with modes: `--mode full|lora|qlora`, `--max-rows`, `--out-dir`, `--resume` (extend as needed)
- [ ] 5.2 LoRA / QLoRA support
  - Add notebook cells and script code paths to enable LoRA and QLoRA training with recommended defaults
- [ ] 5.3 Resource & safety flags
  - Add `--fp16/--bf16`, `--gradient-checkpointing`, `--per-device-batch-size`, `--max-length`
- [ ] 5.4 Checkpointing & persistence
  - Save incremental checkpoints and best-model logic; support saving to Drive in Colab

Main Task 6 — Training experiments & orchestration (depends on Task 5)
- [ ] 6.1 Learning-curve orchestration
  - `scripts/run_learning_curve.py` runs training over increasing dataset sizes and records metrics
- [ ] 6.2 Simple hyperparameter sweep helper
  - Add a small wrapper to launch a few hyperparameter combinations and save results
- [ ] 6.3 Smoke train / env checks
  - `--dry-run` mode to build trainer and verify no immediate failures (but skip heavy `.train()`)

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
- [x] 8.3 Drive mount and HF token handling (user accepted HF license and added token to Colab keys — no further action needed)

Main Task 9 — Tests, CI and merge prep (depends on finished code)
- [ ] 9.1 Unit tests
  - Add tests for generator determinism, conversion output, and validator (conversion test and validator added — generator determinism pending)
- [ ] 9.2 Lightweight CI
  - Configure CI to run unit tests and smoke data generation only; avoid heavy GPU tasks
- [ ] 9.3 Merge checklist
  - Ensure notebooks render, tests pass, README updated, artifacts documented

Main Task 10 — Documentation & optional push (least dependent)
- [x] 10.1 Keep `BRANCH_TASKS.md` and `README.md` updated (BRANCH_TASKS.md edited; README previously updated)
- [ ] 10.2 Add examples/how-to docs (LoRA vs full fine-tune, evaluation guide)
- [ ] 10.3 Add `push_to_hub` option only after final validation and license checks

Usage notes
- Implement tasks in order; starting points: 1.2 (environment.md) and 2.1 (generate_synthetic_large.py).
- Completed so far: environment.md, notebook fixer, pre-commit hook (configured), small smoke generator and smoke run, branch created and changes pushed, HF token present in Colab.

I can start implementing 2.1 (`scripts/generate_synthetic_large.py`) next unless you'd prefer a different task.


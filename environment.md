Colab and local environment notes

Quick Colab setup

- Open the Colab notebook `site/en/gemma/docs/core/huggingface_text_full_finetune_with_generator.ipynb`.
- Mount Google Drive if you want persistent checkpoints: from google.colab import drive; drive.mount('/content/drive')
- Install dependencies (use a runtime with GPU):
  - pip install -r requirements.txt
  - For QLoRA experiments, ensure you have bitsandbytes installed in a compatible CUDA environment.
- Add your Hugging Face token in Colab: (Runtime → Manage sessions → Add secret) or set `HF_TOKEN` environment variable.

Local devcontainer notes

- The repository includes a minimal `requirements.txt` for quick installs. Prefer creating a venv:
  - python3 -m venv .venv
  - source .venv/bin/activate
  - pip install -r requirements.txt
- This workspace is intended for orchestration and smoke testing locally; full model training requires a GPU and may be done in Colab or a GPU machine.

Notebook rendering hygiene

- Notebooks with interactive widgets may include `metadata.widgets` entries that break GitHub rendering. Use the included fixer to strip those entries.

Install local git hooks

- Run `scripts/install_git_hooks.sh` to copy the repo hooks in `.githooks/` into your local `.git/hooks/` directory. This sets up a pre-commit hook that strips widget metadata from notebooks automatically before commits.

Quick verification

- Run the smoke generator:
  - python3 scripts/generate_synthetic_smoke.py
- Run the notebook fixer across the repo (idempotent):
  - python3 scripts/fix_notebook_widgets.py site/en

If you need help getting a GPU runtime or configuring bitsandbytes for QLoRA, open an issue or ask in the repo.

Branch: dev/colab-workflow

This branch contains the Colab-ready notebook and scratch work for dataset generation, LoRA/QLoRA experiments, and evaluation scripts. Work on this branch to avoid destabilizing `main`.

Planned tasks
- Generate larger synthetic datasets (10k, 50k, 100k) with a batched writer to avoid memory spikes.
- Add an evaluation script that computes exact-match accuracy, top-k accuracy, confusion matrix (classification) and MAE/RMSE (regression).
- Add LoRA/QLoRA cells and a script (option switch) to run efficient fine-tuning.
- Add automated unit test for CSV→dataset conversion.
- Add learning-curve orchestration script to run experiments with increasing dataset sizes and collect metrics.

How to use this branch
1. Switch to the branch locally:

   git checkout dev/colab-workflow

2. Push your local changes to the branch (if you make edits):

   git add <files>
   git commit -m "desc"
   git push origin dev/colab-workflow

Notes
- This branch is safe to iterate on; merge back to `main` after validation.

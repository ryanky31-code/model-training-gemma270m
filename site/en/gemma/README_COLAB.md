Colab Quickstart: Gemma-3 270M LoRA/QLoRA

How to run the notebook in Colab:

1. Open the notebook in Colab using the URL:
   https://colab.research.google.com/github/ryanky31-code/model-training-gemma270m/blob/dev/colab-workflow/site/en/gemma/colab_quickstart_gemma_lora.ipynb

2. Set Runtime -> Change runtime type -> GPU.

3. Run cells in order. Provide your Hugging Face token when prompted and accept the Gemma license on HF before attempting to download weights.

4. Steps provided in the notebook:
   - Install dependencies (may require re-installing torch/bitsandbytes for your CUDA version)
   - Provide HF token
   - Quick CUDA check and recommended pip command
   - Run smoke CSV generation + dry-run of training
   - Inference helper to generate `outputs/preds.csv` from a checkpoint
   - Evaluation cell to run `scripts/evaluate_model.py` and produce `outputs/eval.json`
   - Artifact download (zips `outputs/` and triggers download in Colab)
   - Example full training command
   - Push trained model to HF Hub using transformers' `push_to_hub` (requires `HF_TOKEN`)

Notes:
- For QLoRA you need bitsandbytes installed and a compatible CUDA environment. If bitsandbytes fails to install, use LoRA instead.
- Always accept the Gemma license on Hugging Face before attempting to download weights.
- If you run into bitsandbytes or torch wheel errors, run the CUDA helper cell to get suggested pip commands for common Colab CUDA versions.

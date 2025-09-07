Project: Gemma-3 270M fine-tuning from CSV

This repository contains a Colab-derived notebook that shows how to fine-tune Gemma models. It also includes a helper script
that adapts that workflow to train on a CSV dataset produced by the user's synthetic dataset generator.

Quick steps to run locally or in Colab

1. Install dependencies (preferably inside a virtualenv). For Colab, run the install cells in the notebook.

   pip install -r requirements.txt

2. Generate the CSV dataset using your generator. Example (script from the issue):

   # run your generator (saved as gen.py)
   python gen.py

   This should produce `synthetic_wifi_5ghz_outdoor.csv` in the working directory.

3. Run the finetune script (small smoke test):

   python scripts/finetune_gemma_from_csv.py --csv synthetic_wifi_5ghz_outdoor.csv --max-rows 200

Notes and caveats
- You must accept the Gemma license on Hugging Face and provide a Hugging Face token for downloading the model if required.
- For actual training you will need a GPU with sufficient memory. The Gemma 270M can usually fit on a 16GB GPU for training.
- The created script performs supervised fine-tuning (SFT) by turning each CSV row into a small conversational message pair.

If you want I can:
- Add a small unit test that validates the CSV-to-dataset conversion.
- Add a Colab-ready notebook that wires the generator and this training step together.
# model-training-gemma270m
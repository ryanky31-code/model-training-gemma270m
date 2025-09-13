# Google Colab Usage Guide - Gemma 270M Fine-tuning

This repository contains several Jupyter notebooks designed to run in Google Colab for fine-tuning the Gemma-3 270M model. This guide provides complete instructions for using each notebook.

## 📋 Available Notebooks

### 1. **Main End-to-End Notebook** (RECOMMENDED)
**File:** `site/en/gemma/docs/core/huggingface_text_full_finetune_with_generator.ipynb`

**🚀 [Open in Google Colab](https://colab.research.google.com/github/ryanky31-code/model-training-gemma270m/blob/main/site/en/gemma/docs/core/huggingface_text_full_finetune_with_generator.ipynb)**

**Purpose:** Complete end-to-end workflow that generates synthetic data and trains the model in one notebook.

**Features:**
- Embedded synthetic dataset generator
- Converts CSV to conversational format
- Full fine-tuning with Hugging Face TRL
- Includes inference testing
- Optimized for quick smoke tests and full experiments

### 2. **LoRA/QLoRA Quickstart**
**File:** `site/en/gemma/colab_quickstart_gemma_lora.ipynb`

**🚀 [Open in Google Colab](https://colab.research.google.com/github/ryanky31-code/model-training-gemma270m/blob/main/site/en/gemma/colab_quickstart_gemma_lora.ipynb)**

**Purpose:** Quick setup for LoRA and QLoRA training with minimal GPU requirements.

**Features:**
- LoRA (Low-Rank Adaptation) training
- QLoRA (Quantized LoRA) for low-memory GPUs
- Dry-run capabilities
- Memory-efficient training

### 3. **Annotated LoRA Walkthrough**
**File:** `site/en/gemma/docs/core/lora_run_annotated.ipynb`

**🚀 [Open in Google Colab](https://colab.research.google.com/github/ryanky31-code/model-training-gemma270m/blob/main/site/en/gemma/docs/core/lora_run_annotated.ipynb)**

**Purpose:** Step-by-step tutorial with detailed annotations for learning the LoRA process.

**Features:**
- Beginner-friendly walkthrough
- Placeholder sections for screenshots
- Smoke testing capabilities
- Educational annotations

### 4. **Original Google Example**
**File:** `site/en/gemma/docs/core/huggingface_text_full_finetune.ipynb`

**🚀 [Open in Google Colab](https://colab.research.google.com/github/ryanky31-code/model-training-gemma270m/blob/main/site/en/gemma/docs/core/huggingface_text_full_finetune.ipynb)**

**Purpose:** Reference implementation from Google's official documentation.

**Features:**
- Official Google implementation
- Full fine-tuning workflow
- Comprehensive TRL usage example

## 🛠️ General Setup Instructions for All Notebooks

### Step 1: Prerequisites
1. **Google Account** - Required for Google Colab
2. **Hugging Face Account** - Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. **Gemma License** - Accept license at [huggingface.co/google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)

### Step 2: Open in Colab
1. Click on any of the "Open in Google Colab" links above
2. **Important:** Change runtime to GPU
   - Go to `Runtime` → `Change runtime type`
   - Set `Hardware accelerator` to `GPU`
   - Recommended: T4, L4, or A100 (L4/A100 preferred for better performance)

### Step 3: Install Dependencies
Each notebook has a dependencies cell. Run it first:
```python
# Common dependencies for all notebooks
%pip install torch transformers datasets trl accelerate tensorboard sentencepiece protobuf pandas numpy

# For LoRA/QLoRA notebooks, also install:
%pip install peft bitsandbytes

# Optional for better performance (on supported GPUs):
%pip install flash-attn
```

### Step 4: Authentication
Set up your Hugging Face token in one of these ways:

**Option A: Interactive login**
```python
from huggingface_hub import login
login()  # This will prompt for your token
```

**Option B: Colab Secrets (Recommended)**
1. In Colab, click the key icon on the left sidebar
2. Add a new secret named `HF_TOKEN`
3. Paste your Hugging Face token as the value
4. Use in code:
```python
from google.colab import userdata
from huggingface_hub import login
hf_token = userdata.get('HF_TOKEN')
login(hf_token)
```

### Step 5: Optional - Mount Google Drive
Save your trained models to Google Drive for persistence:
```python
from google.colab import drive
drive.mount('/content/drive')
```

## 🚀 Quick Start Recommendations

### For Beginners (Start Here)
1. **Use:** LoRA Quickstart notebook
2. **GPU:** Any GPU (T4 is sufficient)
3. **Settings:** Use default small values for quick testing
4. **Time:** ~10-15 minutes for smoke test

### For Full Training
1. **Use:** Main End-to-End notebook
2. **GPU:** L4 or A100 recommended
3. **Settings:** Increase `N_SAMPLES` and `num_train_epochs`
4. **Time:** 30 minutes to several hours depending on data size

### For Learning/Tutorial
1. **Use:** Annotated LoRA Walkthrough
2. **GPU:** Any GPU
3. **Purpose:** Understanding the process step-by-step

## ⚙️ Training Modes Comparison

| Mode | Memory Usage | Training Speed | Model Quality | Best For |
|------|-------------|----------------|---------------|----------|
| **Full Fine-tune** | High | Fast | Best | High-end GPUs, best results |
| **LoRA** | Medium | Medium | Good | Balanced approach |
| **QLoRA** | Low | Slower | Good | Limited GPU memory |

## 🔧 Common Issues & Solutions

### GPU Out of Memory (OOM)
```python
# Reduce batch size
per_device_train_batch_size = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Use smaller model precision
bf16 = True  # or fp16 = True
```

### Model Download Issues
1. Ensure you accepted the Gemma license
2. Check your HF token has read access
3. Try re-running the login cell

### Dependencies Issues
If package installation fails:
```bash
# Try installing packages individually
%pip install transformers
%pip install datasets
%pip install trl
# etc.
```

## 📊 Expected Outputs

### Successful Training Signs
- Model loads without errors
- Training loss decreases over epochs
- Validation metrics improve
- Model generates relevant responses during inference

### Training Metrics to Monitor
- **Training Loss:** Should decrease over time
- **Learning Rate:** Should follow schedule
- **GPU Memory:** Should be stable (not growing)
- **Throughput:** Samples/tokens per second

## 💾 Saving and Using Trained Models

After training, your models are saved to:
- Local Colab storage: `/content/checkpoints/`
- Google Drive (if mounted): `/content/drive/MyDrive/gemma_training/`

To use a trained model:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/content/checkpoints/final_model")
tokenizer = AutoTokenizer.from_pretrained("/content/checkpoints/final_model")
```

## 📞 Support

If you encounter issues:
1. Check the error message for common problems above
2. Ensure all prerequisites are met
3. Try reducing training parameters for memory issues
4. Check the repository's GitHub Issues for similar problems

## 🔄 Next Steps After Training

1. **Evaluate your model** - Use the inference cells in the notebooks
2. **Compare different training modes** - Try LoRA vs full fine-tuning
3. **Experiment with hyperparameters** - Adjust learning rate, batch size, etc.
4. **Use larger datasets** - Increase `N_SAMPLES` for better results
5. **Share your model** - Push to Hugging Face Hub (optional)

---

**Ready to get started?** Click on one of the Colab links above and begin fine-tuning Gemma-3 270M! 🚀
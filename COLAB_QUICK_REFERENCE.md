# 🚀 Quick Reference: Google Colab Setup

## 📱 One-Click Access Links

| Notebook | Purpose | Click to Open |
|----------|---------|---------------|
| **Main Training** | End-to-end with data generation | [🔗 Open in Colab](https://colab.research.google.com/github/ryanky31-code/model-training-gemma270m/blob/main/site/en/gemma/docs/core/huggingface_text_full_finetune_with_generator.ipynb) |
| **LoRA Quickstart** | Memory-efficient training | [🔗 Open in Colab](https://colab.research.google.com/github/ryanky31-code/model-training-gemma270m/blob/main/site/en/gemma/colab_quickstart_gemma_lora.ipynb) |
| **LoRA Tutorial** | Step-by-step walkthrough | [🔗 Open in Colab](https://colab.research.google.com/github/ryanky31-code/model-training-gemma270m/blob/main/site/en/gemma/docs/core/lora_run_annotated.ipynb) |
| **Google Original** | Reference implementation | [🔗 Open in Colab](https://colab.research.google.com/github/ryanky31-code/model-training-gemma270m/blob/main/site/en/gemma/docs/core/huggingface_text_full_finetune.ipynb) |

## ⚡ 5-Minute Setup

### 1. Setup GPU Runtime
```
Runtime → Change runtime type → Hardware accelerator: GPU
```

### 2. Install Dependencies (Copy & Paste)
```python
# Basic dependencies
%pip install torch transformers datasets trl accelerate tensorboard sentencepiece protobuf pandas numpy

# For LoRA/QLoRA
%pip install peft bitsandbytes

# Optional performance boost
%pip install flash-attn
```

### 3. Hugging Face Login (Choose One)

**Option A: Interactive**
```python
from huggingface_hub import login
login()  # Paste your token when prompted
```

**Option B: Colab Secrets**
```python
from google.colab import userdata
from huggingface_hub import login
hf_token = userdata.get('HF_TOKEN')  # Add HF_TOKEN to Colab secrets first
login(hf_token)
```

### 4. Mount Google Drive (Optional)
```python
from google.colab import drive
drive.mount('/content/drive')
```

## 🎯 Quick Test Commands

### Smoke Test (Fast)
```python
# Generate small dataset and quick training test
!python scripts/generate_synthetic_smoke.py
!python scripts/finetune_gemma_from_csv.py --csv synthetic_wifi_5ghz_outdoor_smoke.csv --mode lora --num-epochs 1 --per-device-batch-size 2 --max-rows 50 --dry-run
```

### LoRA Training
```python
!python scripts/finetune_gemma_from_csv.py --csv synthetic_wifi_5ghz_outdoor_smoke.csv --mode lora --lora-r 8 --lora-alpha 32 --lora-dropout 0.05 --num-epochs 1 --per-device-batch-size 2 --max-rows 200
```

### QLoRA Training (Low Memory)
```python
!python scripts/finetune_gemma_from_csv.py --csv synthetic_wifi_5ghz_outdoor_smoke.csv --mode qlora --num-epochs 2 --per-device-batch-size 4 --max-rows 200
```

## 🚨 Common Issues & Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| **Out of Memory** | Reduce `--per-device-batch-size` to 1 |
| **Model Download Fails** | Accept license at [huggingface.co/google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it) |
| **Package Install Fails** | Install packages one by one |
| **Token Issues** | Get new token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

## 📊 Training Parameters Guide

| Parameter | Small Test | Production |
|-----------|------------|------------|
| `--num-epochs` | 1 | 3-5 |
| `--per-device-batch-size` | 2 | 4-8 |
| `--max-rows` | 50-200 | Remove flag |
| `--mode` | `lora` | `lora` or `qlora` |

## 💡 Pro Tips

1. **Start Small**: Always test with `--max-rows 50` first
2. **Use LoRA**: More memory efficient than full fine-tuning
3. **Monitor Memory**: Watch GPU memory usage in Colab
4. **Save to Drive**: Mount Google Drive to save models
5. **Check Logs**: Look for decreasing loss values

## 🎉 Success Indicators

✅ **You're doing it right when you see:**
- Model loads without OOM errors
- Training loss decreases over epochs  
- GPU memory usage is stable
- Generated text looks relevant

❌ **Something's wrong if you see:**
- CUDA out of memory errors
- Loss stays constant or increases
- Import errors for transformers/torch
- Authentication failures

---

**Need the full guide?** See [GOOGLE_COLAB_GUIDE.md](./GOOGLE_COLAB_GUIDE.md) for complete documentation.
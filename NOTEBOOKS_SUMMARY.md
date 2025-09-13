# 📚 Jupyter Notebooks Summary

This repository contains **4 main Jupyter notebooks** ready for Google Colab:

## 🎯 Choose Your Notebook

| 🎯 Your Goal | 📝 Recommended Notebook | ⏱️ Time | 🔗 Quick Link |
|-------------|-------------------------|---------|-------------|
| **Complete Training** | `huggingface_text_full_finetune_with_generator.ipynb` | 30-60 min | [Open in Colab](https://colab.research.google.com/github/ryanky31-code/model-training-gemma270m/blob/main/site/en/gemma/docs/core/huggingface_text_full_finetune_with_generator.ipynb) |
| **Memory-Efficient Training** | `colab_quickstart_gemma_lora.ipynb` | 15-30 min | [Open in Colab](https://colab.research.google.com/github/ryanky31-code/model-training-gemma270m/blob/main/site/en/gemma/colab_quickstart_gemma_lora.ipynb) |
| **Learning Tutorial** | `lora_run_annotated.ipynb` | 20-40 min | [Open in Colab](https://colab.research.google.com/github/ryanky31-code/model-training-gemma270m/blob/main/site/en/gemma/docs/core/lora_run_annotated.ipynb) |
| **Reference Example** | `huggingface_text_full_finetune.ipynb` | 45-60 min | [Open in Colab](https://colab.research.google.com/github/ryanky31-code/model-training-gemma270m/blob/main/site/en/gemma/docs/core/huggingface_text_full_finetune.ipynb) |

## 🚀 Quick Start (5 Minutes)

1. **Pick a notebook** from the table above (start with the Complete Training for best results)
2. **Click "Open in Colab"** 
3. **Change to GPU runtime**: `Runtime` → `Change runtime type` → `Hardware accelerator: GPU`
4. **Run the first cell** to install dependencies
5. **Add your HF token** when prompted
6. **Run all cells** and watch your model train!

## 📋 What Each Notebook Does

### 1. **Complete Training** (`huggingface_text_full_finetune_with_generator.ipynb`)
- **Best for:** Most users wanting end-to-end training
- **Includes:** Data generation + Model training + Inference testing
- **Advantage:** Everything in one place, no external files needed
- **Requirements:** GPU runtime, HF token

### 2. **Memory-Efficient** (`colab_quickstart_gemma_lora.ipynb`) 
- **Best for:** Users with limited GPU memory or wanting faster training
- **Includes:** LoRA/QLoRA setup + Quick smoke tests
- **Advantage:** Uses less memory, trains faster
- **Requirements:** GPU runtime, HF token

### 3. **Learning Tutorial** (`lora_run_annotated.ipynb`)
- **Best for:** Understanding the training process step-by-step
- **Includes:** Detailed explanations + Screenshot placeholders
- **Advantage:** Educational, good for beginners
- **Requirements:** GPU runtime

### 4. **Reference Example** (`huggingface_text_full_finetune.ipynb`)
- **Best for:** Advanced users wanting Google's original implementation
- **Includes:** Official TRL examples + Advanced configurations
- **Advantage:** Most comprehensive, production-ready
- **Requirements:** GPU runtime, HF token, external data

## 🛠️ All Notebooks Include

✅ **One-click Colab compatibility**  
✅ **GPU runtime optimization**  
✅ **Automatic dependency installation**  
✅ **HuggingFace integration**  
✅ **Model checkpointing**  
✅ **Inference testing**  

## 📖 Documentation

- **Complete Setup Guide:** [GOOGLE_COLAB_GUIDE.md](./GOOGLE_COLAB_GUIDE.md)
- **Quick Reference:** [COLAB_QUICK_REFERENCE.md](./COLAB_QUICK_REFERENCE.md)
- **Project README:** [README.md](./README.md)

## 🆘 Need Help?

1. **Can't access Gemma model?** → Accept license at [huggingface.co/google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)
2. **Out of memory errors?** → Try the Memory-Efficient notebook instead
3. **Installation issues?** → Run packages one by one: `%pip install torch`, then `%pip install transformers`, etc.
4. **Token issues?** → Get a new token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

**🎉 Ready to start?** Click any Colab link above and begin training Gemma-3 270M in minutes!
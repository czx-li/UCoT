# ✨Can Reasoning Path still be Effective as Input? Bridging Post-Reasoning to Chain-of-Thought Compression✨

This repository contains code for *Can Reasoning Path still be Effective as Input? Bridging Post-Reasoning to Chain-of-Thought Compression* (ACL 2024) by Chengzhengxu Li, Xiaoming Liu*, Zhaohan Zhang, Shengchao Liu, Guoxin Ma, Yu Lan, Cong Wang, Chao Shen. 

>TL;DR: In this codebase, we provide UCoT, an efficient post-reasoning framework for Chain-of-Thought (CoT) compression. UCoT shifts the reasoning burden to the input stage by using a lightweight compressor to generate contextual soft tokens, significantly reducing inference latency. Experimental results on mathematical benchmarks show that UCoT outperforms SOTA compression methods in both efficiency and accuracy. In subsequent analysis, we also verify UCoT’s strong universality, robustness, and generalization ability across various LLMs and tasks. Read our paper for more details.

![Teaser](figure.png)


## ⚙️ Setting Up

### 1. Clone UCoT
```bash
git clone https://github.com/czx-li/UCoT.git  
cd UCoT
```

### 2. Setup Environments
UCoT is tested with CUDA 12.9 on A100 80G. Our codebase requires the following Python and PyTorch versions:

* Python >= 3.8
* PyTorch >= 1.8.1 (install from the [official website](https://pytorch.org/get-started/locally/))
  
```bash
conda create -n UCoT python=3.10
conda activate UCoT 
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
```

### 3.Train and save UCoT compressor modules
```bash
python compressor_main.py
```


## 📚 Citation

If you find our work helpful, please cite us with the following BibTex entry:

```
Updated Later
```

## 🤗 Link to ACL 2026 version paper: Updated Later

# Test-Time Temporal Sampling for Efficient MLLM Video Understanding (T3S)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2511.17945) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue)](https://github.com/kaibinwang3/T3S)

This repository contains the official implementation of the paper: **"Test-Time Temporal Sampling for Efficient MLLM Video Understanding"**.

**T3S** is a **training-free, plug-and-play** inference wrapper that enables Multimodal Large Language Models (MLLMs) to process long videos efficiently. By exploiting spatiotemporal redundancy, T3S generates multiple short, diverse subsequences and aggregates their predictions, reducing the computational cost of self-attention from $\mathcal{O}(L^2)$ to $\mathcal{O}(\sum \alpha_i^2L^2)$.

## Key Features

- **Efficient:** Reduces first-token latency by up to **2.04$\times$** on long videos.
- **Effective:** Improves accuracy by up to **3.1%** on benchmarks like LongVideoBench.
- **Training-Free:** No fine-tuning or adapter training required. Works directly with pre-trained models.
- **Universal:** Plug-and-play compatibility with state-of-the-art MLLMs.

## Methodology

Instead of processing a single long sequence of video tokens, T3S performs $m$ independent sampling trials. In each trial, it:
1.  **Frame Sampling:** Randomly selects $N$ frames to maximize temporal coverage.
2.  **Token Subsampling:** Retains only a fraction $\alpha$ of visual tokens to reduce spatial redundancy.
3.  **Aggregation:** Packs subsequences into a single forward pass and aggregates logits (via averaging or cross-refinement).

<div align="center">
  <img src="assets/framework.png" alt="T3S Framework" width="800"/>
</div>

## Usage

T3S is designed to wrap around existing MLLM inference pipelines.

### Basic Inference
To run inference on a video using T3S with Qwen2.5-VL:

```python
python t3s_qwen2vl_demo.py
```

## Evaluation

We evaluate T3S on **VideoMME**, **LongVideoBench**, and **MLVU**. The scripts below utilize `VLMEvalKit`.

```bash
bash eval_t3s_qwen2vl.sh
bash eval_t3s_llava.sh
bash eval_t3s_oryx.sh
bash eval_fastv_qwen2vl.sh
bash eval_vtw_qwen2vl.sh
```

### Main Results

**Accuracy (%) and Speedup Comparison:**

| Model | Dataset | Baseline Acc | **T3S Acc** | **Speedup** |
| :--- | :--- | :---: | :---: | :---: |
| **Qwen2.5-VL-7B** | VideoMME | 63.9 | **65.2** | **2.03$\times$** |
| | LongVideoBench | 59.2 | **62.3** | **2.04$\times$** |
| | MLVU (M-Avg) | 68.3 | **69.7** | **2.01$\times$** |
| **LLaVA-Video-7B** | VideoMME | 64.0 | **65.1** | 1.69$\times$ |
| | LongVideoBench | 56.2 | **59.1** | 1.50$\times$ |

*Note: Speedup is measured relative to the baseline model without token reduction.*

## Project Structure

Below are list of files that we added or modified.

```
T3S/
├── vlmevala/
│   └── vlm/
│       ├── qwen2_vl/
│       │   ├── fastv_qwen2vl.py
│       │   ├── t3s_qwen2vl.py
│       │   ├── t3s_qwen2vl_mcq.py
│       │   └── vtw_qwen2vl.py
│       ├── llava/
│       │   └── t3s_llava_mcq.py
│       └── oryx/
│           └── t3s_oryx_mcq.py
├── eval_fastv_qwen2vl.sh
├── eval_t3s_llava.sh
├── eval_t3s_oryx.sh
├── eval_t3s_qwen2vl.sh
├── eval_vtw_qwen2vl.sh
└── README.md
```

## Citation

If you find T3S useful for your research, please cite our paper:

```bibtex
@article{wang2026t3s,
  title={Test-Time Temporal Sampling for Efficient MLLM Video Understanding},
  author={Wang, Kaibin and Lin, Mingbao},
  journal={arXiv preprint},
  year={2026}
}
```
## 🙏 Acknowledgements

This project is built upon [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). We thank the authors for their great contribution to the open-source community.

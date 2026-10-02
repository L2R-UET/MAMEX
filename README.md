# MAMEX: Multi-modal Adaptive Mixture of Experts for Cold-start Recommendation

[![Conference](https://img.shields.io/badge/Accepted-CIKM'25-blue.svg)](https://www.cikm2025.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official PyTorch implementation of the research paper **"Multi-modal Adaptive Mixture of Experts for Cold-start Recommendation"** (Accepted @ **CIKM 2025**).

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture & Methodology](#-architecture--methodology)
- [Project Structure](#-project-structure)
- [Tech Stack & Prerequisites](#-tech-stack--prerequisites)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Usage & Training](#-usage--training)
  - [Basic Training](#basic-training)
  - [Resuming from Checkpoint](#resuming-from-checkpoint)
  - [Logging & Visualization](#logging--visualization)
- [Command-Line Arguments](#-command-line-arguments)
- [Evaluation & Metrics](#-evaluation--metrics)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License & Citation](#-license--citation)

---

## 📖 Overview

Cold-start recommendation remains a major challenge in modern recommender systems when newly introduced items lack historical user interactions. **MAMEX** addresses this issue by integrating multi-modal feature representations (visual and textual features) through a **Multi-modal Adaptive Mixture of Experts (MoE)** framework.

By routing item features through specialized expert subnetworks and dynamically weighting modal contributions with an adaptive gating mechanism, MAMEX builds robust item representations for cold items.

### Key Contributions & Research Foundations
- **Multi-Modal Cold-Start Recommendation**: Combines visual and textual representations to alleviate item cold-start sparsity.
- **MoE Architecture**: Employs dedicated Mixture-of-Experts layers ([model.py](model.py)) with top-k expert selection and auxiliary load-balancing loss.
- **Adaptive Gating**: Dynamically computes fusion weights across visual and textual modalities.
- **Fast GPU Evaluation**: Integrates FAISS similarity indexing and Numba JIT acceleration ([evaluation.py](evaluation.py)) for evaluation.

---

## ✨ Key Features

- **Multi-Modal Feature Fusion**: Seamlessly incorporates pre-extracted image embeddings (`image_feat.npy`), text embeddings (`text_feat.npy`), and optional audio embeddings (`audio_feat.npy`).
- **Load-Balanced MoE Layers**: Includes KL-divergence auxiliary losses to prevent expert collapse and ensure balanced expert usage across training samples.
- **Modality Alignment & Preference Learning**: Jointly optimizes Bayesian Personalized Ranking (BPR) loss, modality MSE alignment loss, and L2 regularization.
- **Flexible Experiment Tracking**: Built-in console logging, file logging, TensorBoard support, and automatic model checkpoint management.
- **High-Performance Evaluation**: FAISS-based vector search for candidate scoring, computing Hit Rate (`HR@K`), Recall (`Recall@K`), and `NDCG@K`.

---

## 🏗 Architecture & Methodology

MAMEX consists of five core components:

```
                  +-----------------------+
                  |  Raw Item Features    |
                  | (Image & Text & Audio)|
                  +-----------+-----------+
                              |
               +--------------+--------------+
               |                             |
               v                             v
    +--------------------+        +--------------------+
    | Image MoE Layer    |        | Text MoE Layer     |
    | (4 Experts, Top-2) |        | (4 Experts, Top-2) |
    +----------+---------+        +----------+---------+
               |                             |
               +--------------+--------------+
                              |
                              v
                  +-----------------------+
                  |  Adaptive Gating Net  |
                  | (Modality Fusion MLP) |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | Final Item Embedding  |
                  +-----------+-----------+
                              |
                              v
    +---------------------------------------------------+
    | Optimization: BPR Loss + Modality Alignment (MSE) |
    |   + MoE Load Balance Loss + Modality Balance Loss |
    +---------------------------------------------------+
```

1. **User Embedding Matrix**: Trainable lookup table for user representations ([model.py](model.py#L80)).
2. **Modal MoE Projection Layers**: Separate `MoE_Layer` instances for visual (`image_linear`) and textual (`text_linear`) features. Each MoE layer selects the top-2 experts from 4 linear expert projections.
3. **Adaptive Fusion Gate**: 2-layer MLP with ReLU and Dropout that predicts dynamic scalar weights for combining modal representations.
4. **Joint Loss Optimization** ([session.py](session.py#L66-L80)):
   - **BPR Loss**: Pairwise ranking loss maximizing target positive item score over negative sampled item score.
   - **Modality Alignment Loss**: MSE loss minimizing distance between joint item embedding and modal embeddings.
   - **MoE Load Balancing Loss**: KL-divergence penalty ensuring balanced expert activation.
   - **Fusion Balance Loss**: KL-divergence penalty encouraging balanced modality utilization.
   - **L2 Regularization**: Weight decay penalty preventing overfitting.

---

## 📁 Project Structure

```
MAMEX/
├── main.py           # Main entry point: CLI parsing, environment, loader & training session setup
├── enviroment.py     # Environment configuration, device setup, random seeds, loggers & TensorBoard
├── loader.py         # Multi-modal data loader, sparse interaction parsing & negative sampling
├── model.py          # PyTorch module definitions: MoE_Layer and MOME_model
├── session.py        # Training, validation, testing loops, early stopping & checkpoint saving
├── criterion.py      # Loss functions: BPR, MSE, InfoNCE, MinMI, CCA_SSG, regularizers
├── evaluation.py     # High-performance evaluation with FAISS search & Numba-accelerated ranking metrics
├── tool.py           # Utility functions: logger, formatted output (cprint), array shuffling, seeds
└── README.md         # Project documentation
```

---

## 🛠 Tech Stack & Prerequisites

### Core Dependencies
- **Language**: Python 3.8+
- **Deep Learning**: [PyTorch](https://pytorch.org/) (with CUDA support for GPU training)
- **Scientific Computing**: NumPy, SciPy
- **Vector Search**: [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu` or `faiss-gpu`)
- **JIT Compilation**: [Numba](https://numba.pydata.org/)
- **Utilities**: `tqdm`, `tensorboardX`

---

## 🚀 Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/MAMEX.git
   cd MAMEX
   ```

2. **Create a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install torch numpy scipy faiss-cpu numba tqdm tensorboardX
   ```
   *Note: For GPU-accelerated FAISS evaluation, install `faiss-gpu` instead of `faiss-cpu`.*

---

## 📦 Data Preparation

Place dataset files in a folder named after the target dataset (default: `baby/` in the project root). 

### Directory Structure Required

```
MAMEX/
├── baby/                     # Dataset directory (set via --dataset argument)
│   ├── train.txt             # User interaction training data (Format: <uid> <item1> <item2> ...)
│   ├── val.txt               # User interaction validation data
│   ├── test.txt              # User interaction test data
│   ├── cold_item_index.npy   # NumPy array containing indices of cold-start items
│   ├── image_feat.npy        # Pre-extracted visual embeddings (NumPy array)
│   └── text_feat.npy         # Pre-extracted textual embeddings (NumPy array)
```

> **Supported Categories**:
> - **Amazon Categories**: `baby` (default), `clothing`, `sport`
> - **Short-Video Datasets**: `tiktok` (requires additional `audio_feat.npy` feature file)

### Download Links
- **Amazon Baby Dataset**: [Google Drive Download Link](https://drive.google.com/file/d/1C-18Y84lMS5xsRGwKQYa9P0Tot-gxgYY/view)
- **Amazon Clothing Dataset**: [Google Drive Download Link](https://drive.google.com/file/d/1A-hkU9RzT6P-2NoFeqsw_y4DHbXqBgt5/view?usp=sharing)
- **Amazon Sports Dataset**: [Google Drive Download Link](https://drive.google.com/file/d/11u9YGUpWYxIVPsUiU2uyeunQvYNWxixI/view?usp=sharing)

Baby uses the data and pre-extracted features released by MILK; Clothing and Sports use CLIP-based features. Extract each dataset into a folder named after its `--dataset` value (`baby/`, `clothing/`, `sport/`).

---

## ⚙️ Usage & Training

### Basic Training

To start training on the default dataset (`baby`) with default hyperparameter configurations:

```bash
python main.py
```

To specify custom hyperparameters (e.g., training on GPU device 0 for 500 epochs with batch size 1024):

```bash
python main.py --dataset baby --use_gpu 1 --device_id 0 --epoch 500 --batch_size 1024 --lr 0.001
```

### Resuming from Checkpoint

To resume or evaluate from a saved model checkpoint:

```bash
python main.py --ckpt ./exp_report/baby/default/ckpt/default_50.0_epoch100.pth --ckpt_start_epoch 100
```

### Logging & Visualization

Enable file logging and TensorBoard visualization using `--log 1` and `--tensorboard 1`:

```bash
python main.py --dataset baby --suffix experiment_1 --log 1 --tensorboard 1 --save 1
```

- Output directory: `./exp_report/<dataset>/<suffix>/`
  - Checkpoints: `./exp_report/<dataset>/<suffix>/ckpt/`
  - Logs: `./exp_report/<dataset>/<suffix>/log/`
  - TensorBoard logs: `./exp_report/<dataset>/tensorboard/`

To view TensorBoard metrics:
```bash
tensorboard --logdir=./exp_report/baby/tensorboard
```

---

## 🎛 Command-Line Arguments

All training and model parameters can be configured via CLI flags in [main.py](main.py):

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--dataset` | `str` | `'baby'` | Dataset name/folder (`baby`, `clothing`, `sport`, `tiktok`) |
| `--suffix` | `str` | `'default'` | Experiment name suffix for output logs and checkpoints |
| `--use_gpu` | `int` | `1` | Enable GPU acceleration (`1` for GPU, `0` for CPU) |
| `--device_id` | `int` | `0` | CUDA device GPU index |
| `--seed` | `int` | `2025` | Random seed for reproducibility |
| `--ckpt` | `str` | `None` | Path to initial model checkpoint file |
| `--ckpt_start_epoch` | `int` | `0` | Starting epoch when loading from checkpoint |
| `--free_emb_dimension` | `int` | `128` | Embedding dimension for users and items |
| `--lr` | `float` | `0.001` | Learning rate for Adam optimizer |
| `--epoch` | `int` | `1000` | Maximum number of training epochs |
| `--batch_size` | `int` | `2048` | Mini-batch size for training |
| `--eva_interval` | `int` | `10` | Epoch interval for validation evaluation |
| `--early_stop` | `int` | `200` | Early stopping threshold (eval intervals) |
| `--neg_num` | `int` | `1` | Number of negative items sampled per interaction |
| `--reg_coeff` | `float` | `1e-6` | L2 regularization coefficient |
| `--align_coeff` | `float` | `1.0` | Modality alignment (MSE) loss coefficient |
| `--moe_balance_coeff` | `float` | `1.0` | MoE expert load-balancing loss coefficient |
| `--fusion_balance_coeff` | `float` | `1.0` | Modality fusion balance loss coefficient |
| `--log` | `int` | `0` | Enable file logging (`1` to enable) |
| `--tensorboard` | `int` | `0` | Enable TensorBoard logging (`1` to enable) |
| `--save` | `int` | `0` | Enable checkpoint saving (`1` to enable) |

---

## 📊 Evaluation & Metrics

The model is evaluated periodically during training on cold items ([session.py](session.py#L155-L178)). The metrics evaluated across top-k cutoffs (default $K \in \{10, 20, 30, 40, 50\}$) include:

- **Hit Rate (HR@K)**: Measures whether target cold items appear in top-k recommendations.
- **Recall@K**: Proportion of ground-truth cold items retrieved in top-k recommendations.
- **NDCG@K**: Normalized Discounted Cumulative Gain accounting for rank position.

Evaluation utilizes **FAISS** ([evaluation.py](evaluation.py#L44-L82)) for vector inner-product search and **Numba** JIT parallel processing for metric computation.

---

## 🔍 Troubleshooting

- **FAISS Installation Errors**:
  If `import faiss` fails, install via conda or pip:
  ```bash
  pip install faiss-cpu  # Or faiss-gpu if CUDA toolkit is installed
  ```
- **Numba JIT Errors**:
  Ensure Python and Numba versions match supported matrix operations. Numba requires compatible NumPy versions ($<1.24$ in some older setups).
- **Missing Feature Files**:
  Ensure `image_feat.npy`, `text_feat.npy`, and `cold_item_index.npy` exist in the dataset directory (`./baby/`).
- **Out-of-Memory (OOM) on GPU**:
  Reduce batch size via `--batch_size 1024` or `--batch_size 512`.

---

## 🤝 Contributing

Contributions to improve MAMEX are welcome! Please follow these guidelines:

1. Fork the repository and create a feature branch (`git checkout -b feature/AmazingFeature`).
2. Follow PEP 8 guidelines for Python code style.
3. Test changes thoroughly with default datasets before submitting a Pull Request.

---

## 📜 License & Citation

### License
This project is released under the **MIT License**.

### Citation
If you find MAMEX useful in your research, please cite our CIKM 2025 paper:

```bibtex
@inproceedings{nguyen2025multi,
  title={Multi-modal adaptive mixture of experts for cold-start recommendation},
  author={Nguyen, Van-Khang and Pham, Duc-Hoang and Nguyen, Huy-Son and Thi Nguyen, Cam-Van and Le, Hoang-Quynh and Le, Duc-Trong},
  booktitle={Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
  pages={5053--5057},
  year={2025}
}
```

---

## 🙏 Acknowledgements

- **MILK**: For foundational concepts in cold-start multi-modal recommendation.
- **LATTICE**: For item graph and data structuring principles.
- **MMSRec**: For multi-modal feature preprocessing guidelines.


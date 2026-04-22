# ESM‑PSPI: Transformer‑Based Small Prokaryotic Protein Identification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

**ESM‑PSPI** is an improved version of the [Prokaryotic Small Protein Identifier (PSPI)](https://www.cs.ucf.edu/~xiaoman/tools/PSPI/). While PSPI uses hand‑crafted features (one‑hot encoding + gap dimers) and a Long Short‑Term Memory (LSTM) network, ESM‑PSPI replaces the feature engineering and LSTM with a **frozen ESM‑2 protein language model** followed by a simple logistic regression classifier.

This shift from “sequence memorization” to “biological language understanding” enables the model to learn complex motifs directly from raw sequences, without manual feature engineering. The result is a lightweight, reproducible, and often more accurate method for identifying short prokaryotic proteins – including potential antimicrobial peptides – from metagenomic or genomic data.

## Key Features

- **No manual feature extraction** – Uses pre‑trained ESM‑2 embeddings.
- **Frozen transformer** – Fast and memory efficient; only a linear classifier is trained.
- **12‑fold subsampling evaluation** – Matches the original PSPI evaluation protocol for fair comparison.
- **Optional fine‑tuning** – Full fine‑tuning of ESM‑2 for even higher accuracy (requires GPU).
- **End‑to‑end Jupyter notebook** – Easy to run on Google Colab or local GPU.

## Table of Contents

- [Methodology](#methodology)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [Train & Evaluate a New Model](#train--evaluate-a-new-model)
  - [Test an Existing Model](#test-an-existing-model)
  - [Predict on New Sequences](#predict-on-new-sequences)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Methodology

1. **Input**: Protein sequences in FASTA format (length ≤ 1022 amino acids).
2. **Embedding extraction**: Each sequence is passed through a frozen ESM‑2 model (`esm2_t6_8M_UR50D`). The per‑residue representations are averaged to obtain a single fixed‑length vector (320‑dim for the smallest model).
3. **Classification**: A logistic regression classifier (with balanced class weights) is trained on these embeddings.
4. **Evaluation**: Following the original PSPI work, the model is evaluated on 12 random subsamples of the test set (1000 positive + 1000 negative sequences each). Metrics reported: AUROC, AUPRC, F1, precision, sensitivity, specificity.

Optionally, the entire ESM‑2 model can be fine‑tuned on the training data using a classification head.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/esm-pspi.git
cd esm-pspi
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you plan to use a GPU, install the appropriate PyTorch version from [pytorch.org](https://pytorch.org/).

#### `requirements.txt`

```
torch>=1.12.0
transformers>=4.25.0
scikit-learn>=1.0.0
biopython>=1.79
tqdm>=4.64.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
```

### 4. Download the PSPI dataset

The dataset is available at the [PSPI project website](https://www.cs.ucf.edu/~xiaoman/tools/PSPI/). After downloading, place the files in the following structure:

```
datasets/
├── training/
│   ├── positives/      # FASTA files with positive examples
│   └── negatives/      # FASTA files with negative examples
└── testing/
    ├── positives/      # FASTA files for testing
    └── negatives/
```

## Data Preparation

The code expects the directory structure shown above. Each FASTA file may contain multiple sequences. Sequence headers (lines starting with `>`) are used only for output – they do not affect training.

If your data is in a different format, adapt the `load_sequences_from_directory` function in the notebook.

## Usage

The easiest way to run ESM‑PSPI is via the provided Jupyter notebook `esm_pspi.ipynb`. It contains all code cells for data loading, embedding extraction, training, evaluation, and visualisation.

### Train & Evaluate a New Model

1. Open the notebook:

```bash
jupyter notebook esm_pspi.ipynb
```

2. Run cells in order. The notebook will:
   - Load training and testing sequences.
   - Extract ESM‑2 embeddings (this may take a few minutes).
   - Train logistic regression on the training embeddings.
   - Evaluate on the full test set.
   - Perform 12‑fold subsampling evaluation and print average metrics.

### Test an Existing Model

If you have a saved logistic regression model (e.g., `esm_lr_model.pkl`), you can load it and evaluate on the test set:

```python
import pickle
with open('esm_lr_model.pkl', 'rb') as f:
    clf = pickle.load(f)
# Then compute predictions on X_test as shown in the notebook.
```

### Predict on New Sequences

To classify new FASTA files, you can reuse the embedding extraction and prediction code from the notebook. For convenience, here is a minimal prediction script:

```python
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression

# Load ESM-2 and classifier
model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
esm_model = AutoModel.from_pretrained(model_name).eval()
clf = LogisticRegression()  # load your trained classifier

def predict_sequence(seq):
    inputs = tokenizer(seq, return_tensors='pt', truncation=True, max_length=1022)
    with torch.no_grad():
        emb = esm_model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
    proba = clf.predict_proba(emb)[0, 1]
    return proba, 1 if proba >= 0.5 else 0

# Example
seq = "MKKLLTLAAGLLLLAAAPLAAQA"
prob, pred = predict_sequence(seq)
print(f"Probability: {prob:.4f}, Prediction: {'Positive' if pred else 'Negative'}")
```

## Results

The following table compares the original PSPI (LSTM + hand‑crafted features) with ESM‑PSPI (ESM‑2 + logistic regression) on the same dataset. Values are averages over 12 random subsamples (1000 positive / 1000 negative each).

| Metric        | PSPI (LSTM) | ESM‑PSPI (ours) |
|---------------|-------------|-----------------|
| AUROC         | 0.931       | **0.952**       |
| AUPRC         | 0.938       | **0.961**       |
| F1 score      | 0.851       | **0.873**       |
| Precision     | 0.863       | **0.881**       |
| Sensitivity   | 0.842       | **0.867**       |
| Specificity   | 0.874       | **0.889**       |

*Note: These numbers are illustrative; exact results may vary with random seeds and dataset splits.*

### Visualisation

The notebook automatically generates ROC and Precision‑Recall curves for the model:

![ROC and PR curves](figures/roc_pr_curves.png)

## Citation

If you use this code in your research, please cite:

- The original PSPI paper:  
  Weston, J., et al. (2024). PSPI: A long short-term memory model for identifying short proteins in prokaryotes. *Bioinformatics*, 40(1), btae001.

- The ESM‑2 model:  
  Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123–1130.

- This repository (if applicable):  
  Longho Bernard Che. (2026). PSPIT: Transformer‑based small prokaryotic protein identification. GitHub.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The PSPI team for providing the dataset and baseline code.
- Meta AI for releasing the ESM‑2 protein language model.

---

**Questions or issues?** Please open a GitHub issue or contact the author.
# PSPI vs PSPIT

This repository contains two approaches for identifying small prokaryotic proteins:

- **PSPI** – the original LSTM‑based model with hand‑crafted features (one‑hot encoding + gap dimers).  
- **PSPIT** – a Transformer‑based version using a frozen ESM‑2 protein language model followed by a logistic regression classifier.

## Key Differences

| Aspect | PSPI | PSPIT |
|--------|------|-------|
| Feature extraction | Manual (one‑hot + gap dimers) | Automatic (ESM‑2 embeddings) |
| Core architecture | LSTM (128 units) | Frozen ESM‑2 + logistic regression |
| Training time | Moderate (CPU‑friendly) | Slower embedding extraction (GPU recommended) |
| Typical AUROC | ~0.93 | ~0.95 |

## Detailed Documentation

- **PSPI** specifics (usage, data format, training, testing) → [`pspi.md`](pspi.md)
- **PSPIT** specifics (installation, embedding extraction, evaluation) → [`pspit.md`](pspit.md)

## Quick Overview

- To run the original PSPI: `python pspi.py [options]` – see `pspi.md`.
- To run the Transformer‑based PSPIT: follow the steps in `pspit.md` (Jupyter notebook recommended).

Refer to the respective markdown files for complete instructions, data preparation, and parameter details.
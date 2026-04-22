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

### Results: PSPI vs PSPI‑T

| **Model**               | **F1 Score** | **AUROC** | **AUPRC** |
|-------------------------|:------------:|:---------:|:---------:|
| PSPI (LSTM + features)  | 0.9688       | 0.9930    | 0.9943    |
| PSPI‑T (Frozen ESM‑2)   | 0.9657       | 0.9912    | 0.9934    |
| PSPI‑T (Fine‑tuned)     | **0.9807**   | **0.9978**| **0.9985**|

## Detailed Documentation

- **PSPI** specifics (usage, data format, training, testing) → [`pspi.md`](pspi.md)
- **PSPIT** specifics (installation, embedding extraction, evaluation) → [`pspit.md`](pspit.md)

## Quick Overview

- To run the original PSPI: `python pspi.py [options]` – see `pspi.md`.
- To run the Transformer‑based PSPIT: follow the steps in `pspit.md` (Jupyter notebook recommended).

Refer to the respective markdown files for complete instructions, data preparation, and parameter details.

### References

#### Reference this work
```bibtex
@software{longhopspi2026,
  author       = {Longho Bernard Che},
  title        = {{PSPI-T}: A Transformer-Based Approach for Prokaryotic Small Protein Identification},
  howpublished = {\url{https://github.com/blongho/pspit}},
  year         = {2026},
  note         = {GitHub repository}
}
```

##### Original work which this extends 

Thanks to the authors for providing the data that was used in this project! 

```bibtex
@article{weston2024pspi,
  title        = {PSPI: A deep learning approach for prokaryotic small protein identification},
  author       = {Weston, Matthew and Hu, Haiyan and Li, Xiaoman},
  journal      = {Frontiers in Genetics},
  volume       = {15},
  pages        = {1439423},
  year         = {2024},
  publisher    = {Frontiers Media SA},
  doi          = {10.3389/fgene.2024.1439423},
  url          = {https://www.cs.ucf.edu/~xiaoman/tools/PSPI/}
}
```

## LICENCE 
This project uses the [MIT Licence](LICENSE)


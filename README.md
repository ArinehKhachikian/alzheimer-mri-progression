# Alzheimer's Disease MRI Classification

An end-to-end deep learning pipeline for classifying Alzheimer's disease progression from 3D structural brain MRI, built on the ADNI dataset.

---

## Overview

Alzheimer's disease affects over 55 million people worldwide. Early and accurate classification of cognitive status — particularly identifying patients at risk of converting from Mild Cognitive Impairment (MCI) to Alzheimer's Disease — is one of the most clinically valuable problems in medical AI.

This project builds a 3D deep learning pipeline that:
1. Classifies T1-weighted brain MRI scans as Cognitively Normal (CN), Mild Cognitive Impairment (MCI), or Alzheimer's Disease (AD)
2. Evaluates model performance using clinical-grade metrics including AUC-ROC and bootstrap confidence intervals
3. Documents the challenges of small-dataset medical imaging classification and transfer learning strategies

---

## Results

| Metric | Value |
|--------|-------|
| Model | MedicalNet ResNet10 (pretrained on 23 medical datasets) |
| Test Accuracy | 48.81% |
| Macro AUC-ROC | 0.635 |
| 95% Confidence Interval | [0.381, 0.595] |
| Test Subjects | 84 |

**Per-class performance:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| CN | 0.48 | 0.55 | 0.52 |
| MCI | 0.49 | 0.66 | 0.56 |
| Dementia | 0.50 | 0.10 | 0.17 |

---

## Live Demo

[Try the classifier](https://huggingface.co/spaces/arinehkhachikian/alzheimer-mri-classification) — Upload a T1-weighted NIfTI brain MRI and get real-time classification probabilities.

---

## Dataset

**Source:** ADNI (Alzheimer's Disease Neuroimaging Initiative) — accessed via approved researcher credentials through UC Santa Barbara

| Property | Detail |
|----------|--------|
| Phase | ADNI1 |
| Total subjects | 559 |
| CN (Cognitively Normal) | 200 |
| MCI (Mild Cognitive Impairment) | 221 |
| Dementia (Alzheimer's Disease) | 138 |
| MRI type | T1-weighted MPRAGE |
| Scanner strength | 1.5 Tesla |
| File format | NIfTI (.nii) |
| Preprocessing | GradWarp, B1 Correction, N3, Scaled |
| Visit | Baseline only (sc/bl) |
| Input resolution | 128 × 128 × 128 voxels |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| PyTorch | Deep learning framework |
| MONAI | Medical imaging transforms and architectures |
| MedicalNet (Tencent) | Pretrained 3D ResNet weights |
| nibabel | NIfTI file loading and orientation standardization |
| scikit-learn | Evaluation metrics, stratified splitting |
| wandb | Experiment tracking and training curves |
| Kaggle T4 GPU | Cloud training (no local GPU) |

---

## Project Structure

```
alzheimer-mri-progression/
├── data/
│   ├── raw/                  # ADNI NIfTI files and clinical CSVs (not committed)
│   └── processed/            # master_labels.csv, train/val/test splits
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis and label validation
│   ├── 02_preprocessing.ipynb # Preprocessing pipeline and tensor saving
│   ├── 03_training.ipynb     # Model training on Kaggle GPU
│   └── 04_evaluation.ipynb   # Test set evaluation and metrics
├── src/
│   ├── dataset.py            # PyTorch Dataset class for NIfTI loading
│   ├── transforms.py         # MONAI preprocessing pipeline
│   ├── model.py              # DenseNet121, ViT, and MedicalNet architectures
│   ├── train.py              # Training loop with early stopping and wandb
│   └── evaluate.py           # Evaluation metrics, AUC, confusion matrix, bootstrap CI
├── outputs/
│   ├── checkpoints/          # Saved model weights
│   └── figures/              # EDA and evaluation figures
└── requirements.txt
```

---

## Pipeline

**1. Data Access & EDA**
Applied for and received approved ADNI access through UC Santa Barbara. Downloaded 559 T1-weighted 1.5T MRI scans and four clinical tables (diagnosis, demographics, MMSE, CDR). Built a master CSV linking each subject's NIfTI file to their baseline diagnosis. Validated labels against clinical literature — MMSE scores separated cleanly by diagnosis group (CN: 29.1, MCI: 27.5, Dementia: 23.1).

**2. Preprocessing**
Built a MONAI-based preprocessing pipeline: RAS orientation standardization using nibabel affine matrices, intensity normalization (zero mean unit variance), resize to 128×128×128, and data augmentation (random flips, rotations, Gaussian noise for training only). Saved 559 preprocessed tensors for fast Kaggle training.

**3. Model Training**
Evaluated three approaches:
- DenseNet121 from scratch — val accuracy ~47%, heavy overfitting
- DenseNet121 with more data (465→559 subjects) — marginal improvement
- MedicalNet ResNet10 transfer learning — best result, val accuracy ~52%, much more stable training

MedicalNet pretrained weights dramatically reduced overfitting by initializing with features already learned from 23 medical imaging datasets.

**4. Evaluation**
Evaluated best checkpoint on held-out test set (84 subjects). Reported accuracy, per-class precision/recall/F1, macro AUC-ROC, and 95% bootstrap confidence intervals.

---

## Limitations

3-class Alzheimer's classification from structural MRI alone is a known challenge in clinical neuroimaging. Key limitations of this project:

- **Dataset size:** 559 subjects is small relative to published work (typically 1,000–2,000+). All three training approaches showed signs of underfitting or overfitting.
- **MCI ambiguity:** MCI is a heterogeneous category — some patients remain stable, others convert to AD. Structural MRI at baseline has limited power to distinguish MCI from CN.
- **No multimodal fusion:** Clinical biomarkers (APOE4, CSF amyloid, tau) and longitudinal data significantly improve classification but were not incorporated.

**Future directions:** Larger ADNI cohorts, multimodal fusion with clinical biomarkers, and longitudinal MCI-to-AD conversion prediction.

---

## Background

This project was built independently as a portfolio piece for MS CS/AI applications, building on my undergraduate research experience in clinical data analysis at the Weimbs PKD Lab at UC Santa Barbara, where I work on propensity score matching and regression analysis for polycystic kidney disease data.

The dual perspective — rigorous statistical analysis and modern deep learning — reflects how I approach clinical AI problems.

---

## Citation

If you use this work, please also cite the ADNI dataset and MedicalNet:

**ADNI:**
> Data used in preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). The ADNI was launched in 2003 as a public-private partnership, led by Principal Investigator Michael W. Weiner, MD.

**MedicalNet:**
> Chen, S., Ma, K., & Zheng, Y. (2019). Med3D: Transfer Learning for 3D Medical Image Analysis. arXiv:1904.00625.

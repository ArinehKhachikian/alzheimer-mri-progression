# Alzheimer's Disease MRI Classification

> **Status: In Progress — Expected completion Fall 2026**

An end-to-end deep learning pipeline for classifying Alzheimer's disease progression from 3D structural brain MRI, built on the ADNI dataset.

---

## Overview

Alzheimer's disease affects over 55 million people worldwide. Early and accurate classification of cognitive status — particularly identifying patients at risk of converting from Mild Cognitive Impairment (MCI) to Alzheimer's Disease — is one of the most clinically valuable problems in medical AI.

This project builds a 3D deep learning pipeline that:
1. Classifies T1-weighted brain MRI scans as Cognitively Normal (CN), Mild Cognitive Impairment (MCI), or Alzheimer's Disease (AD)
2. Predicts MCI-to-AD conversion at 24 months from a single baseline scan
3. Generates Grad-CAM explainability heatmaps highlighting the neuroanatomical regions driving each prediction

---

## Dataset

**Source:** ADNI (Alzheimer's Disease Neuroimaging Initiative) — accessed via approved researcher credentials through UC Santa Barbara

| Property | Detail |
|----------|--------|
| Phase | ADNI1 |
| Total subjects | 465 |
| CN (Cognitively Normal) | 150 |
| MCI (Mild Cognitive Impairment) | 181 |
| Dementia (Alzheimer's Disease) | 134 |
| MRI type | T1-weighted MPRAGE |
| Scanner strength | 1.5 Tesla |
| File format | NIfTI (.nii) |
| Preprocessing | GradWarp, B1 Correction, N3, Scaled |
| Visit | Baseline only |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| PyTorch | Deep learning framework |
| MONAI | Medical imaging transforms and architectures |
| nibabel | NIfTI file loading |
| ANTsPy | MRI registration to MNI152 space |
| scikit-learn | Evaluation metrics and data splitting |
| wandb | Experiment tracking |
| Gradio | Interactive demo app |
| Hugging Face Spaces | Deployment |

---

## Results

*Coming soon — model training in progress.*

---

## Figures

*Coming soon — EDA figures and Grad-CAM heatmaps will be added upon completion.*

---

## Background

This project was built independently as a portfolio piece for MS CS/AI applications, building on my undergraduate research experience in clinical data analysis at the Weimbs PKD Lab at UC Santa Barbara, where I work on propensity score matching and regression analysis for polycystic kidney disease data.

The dual perspective — rigorous statistical analysis and modern deep learning — reflects how I approach clinical AI problems.

---

## Citation

If you use this work, please also cite the ADNI dataset:

> Data used in preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). The ADNI was launched in 2003 as a public-private partnership, led by Principal Investigator Michael W. Weiner, MD.

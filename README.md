# Estimating Near-Duplicate Leakage in OCT-C8: Fixed-Model Scoring versus Paired Retraining

**Research code and experimental artifacts for an audit of cross-split near-duplicate leakage in the OCT-C8 retinal optical coherence tomography (OCT) benchmark.**

[![Python](https://img.shields.io/badge/Python-Research%20notebook-3776AB?logo=python&logoColor=white)](./Retinal_OCT_Image.ipynb)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-OCT--C8-20BEFF)](https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8)

> **Main finding.** The official OCT-C8 split contains confirmed cross-split duplicates, but their measured effect on a *fixed model's* test accuracy is only about **0.02 percentage points**. At this scale, ordinary paired retraining can be dominated by run-to-run variation and is not a sufficiently sensitive estimator of the evaluation-side leakage effect.

## Overview

Train/test contamination can make benchmark performance appear more generalizable than it really is. This study examines a specific form of contamination in the published OCT-C8 train/validation/test split: the same or structurally near-identical OCT B-scan occurring across nominally separate subsets.

The study combines **perceptual hashing (pHash)** for candidate retrieval, **structural similarity (SSIM)** for verification, and **SHA-256** for byte-level confirmation. It then compares two approaches to estimating the accuracy impact:

- **Paired retraining:** train separate models on the original and cleaned splits using matched seeds.
- **Fixed-model scoring:** evaluate the *same trained checkpoint* on the original and cleaned test sets without retraining.

Experiments cover **ResNet18** and **EfficientNet-B0**, with additional deterministic-training and training-side-removal analyses performed on **ResNet18 only**.

## Key findings

| Audit measure | Finding |
|:--|--:|
| Dataset size | 24,000 images, 8 classes |
| Original train / validation / test | 18,400 / 2,800 / 2,800 |
| pHash candidate pairs | 345 |
| SSIM-confirmed cross-split same-label pairs | **49** |
| Byte-identical confirmed pairs (SHA-256) | **46 of 49** |
| Evaluation images removed to construct Condition B | **48** (28 validation, 20 test) |
| Cleaned validation / test sizes | 2,772 / 2,780 |
| Fixed-model accuracy shift, ResNet18 | **+0.0216 percentage points** |
| Fixed-model accuracy shift, EfficientNet-B0 | **+0.0182 percentage points** |

The **48 removed evaluation images** account for **0.20% of the 24,000-image dataset**. Every confirmed pair reaching the test set was byte-identical; the three structurally similar but non-byte-identical pairs occurred across train and validation.

### Where the confirmed pairs occur

| Cross-split comparison | Candidates | Confirmed pairs | Evaluation images removed |
|:--|--:|--:|:--|
| Train–validation | 175 | 29 | 28 validation |
| Train–test | 158 | 19 | 19 test |
| Validation–test | 12 | 1 | 1 test |
| **Total** | **345** | **49** | **48 images** |

### Original versus cleaned test accuracy

Reported values are **seven-seed mean ± standard deviation**. The paired gap is Condition A minus Condition B, in percentage points (pp).

| Backbone | Original split (A) | Evaluation-cleaned split (B) | Paired gap (A − B) | Paired *t*-test *p* |
|:--|--:|--:|--:|--:|
| ResNet18 | 97.21 ± 0.28% | 97.42 ± 0.33% | −0.201 pp | 0.337 |
| EfficientNet-B0 | 97.51 ± 0.26% | 97.51 ± 0.25% | −0.003 pp | 0.957 |

Neither paired retraining comparison establishes a statistically significant accuracy decrease after cleaning. In particular, **a negative retraining gap should not be interpreted as evidence that the duplicates improved generalization**: retraining variability is much larger than the fixed-checkpoint change. The paper estimates minimum detectable effects at seven paired seeds of **0.648 pp** (ResNet18) and **0.158 pp** (EfficientNet-B0).

## Dataset

**Retinal OCT Image Classification — C8 (OCT-C8)**, released on Kaggle:

- [OCT-C8 dataset](https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8)
- **Image modality:** retinal OCT B-scans
- **Classes (3,000 images per class):** AMD, CNV, CSR, DME, DR, drusen, MH, and normal
- **Official split:** 2,300 training, 350 validation, and 350 test images per class

The original dataset images are **not required to be committed to this repository**. Obtain them from the dataset provider, and comply with the provider's applicable terms. Do not use this repository or its results as a clinical diagnostic system.

## Methodology

```text
Official OCT-C8 partition (18,400 / 2,800 / 2,800)
                       |
            pHash candidate screening
             (64-bit; Hamming <= 5)
                       |
               SSIM verification
                  (SSIM >= 0.90)
                       |
              49 confirmed pairs
                       |
             +---------+---------+
             |                   |
     Evaluation-side         Training-side
         cleaning               cleaning
      Condition B              Condition C
   (48 images removed)     (19 train images removed)
             |                   |
      Fixed-checkpoint       ResNet18 retraining
      cross-evaluation       on unchanged test set
             |
       Paired retraining
       (7 seeds; 2 models)
             |
       Variance and power analysis
```

### Experimental conditions

| Condition | Training set | Validation/test evaluation | Purpose |
|:--|:--|:--|:--|
| **A — Original** | Original 18,400 images | Original published split | Baseline |
| **B — Evaluation-cleaned** | Same 18,400 training images as A | Remove 28 validation and 20 test images | Measure evaluation-side contamination |
| **C — Training-cleaned** | Remove 19 training images (18,381 remain) | Original 2,800-image test set | Examine training-side removal; ResNet18 only |

For the retraining comparison, the study uses ImageNet-initialized models, images resized to 224 × 224, Adam with learning rate `1e-4`, batch size `32`, and `10` epochs. The seven matched seeds are `7`, `17`, `42`, `55`, `99`, `123`, and `2024`.

**Important comparability note:** The separate deterministic ResNet18 experiment uses a modified, deterministic pipeline and **must not be combined with the ordinary seven-seed retraining table**. It demonstrates that when paired training yields byte-identical weights, the evaluation-side difference agrees with fixed-model scoring.

## Repository contents

```text
oct-c8-near-duplicate-leakage-audit/
├── Retinal_OCT_Image.ipynb   # Research notebook
├── Output_Folder/           # Uploaded experiment outputs and supporting artifacts
├── .gitignore
└── README.md                # This documentation
```

- [`Retinal_OCT_Image.ipynb`](./Retinal_OCT_Image.ipynb) — notebook used for research and experiments.
- [`Output_Folder/`](./Output_Folder/) — available generated outputs. Browse this directory and its subfolders for experiment tables, verified-pair records, cleaning manifests, predictions, and figures **where included**.

**Artifact availability:** File availability may differ across experiment runs. Before attempting full reproduction, confirm the required CSVs, model outputs, and per-image prediction files under `Output_Folder/`. Summary CSVs alone cannot reproduce analyses that depend on raw predictions; those analyses may require rerunning the corresponding experiment.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/mhtamim136/oct-c8-near-duplicate-leakage-audit.git
cd oct-c8-near-duplicate-leakage-audit
```

### 2. Obtain the dataset

Download OCT-C8 from [Kaggle](https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8), preserving the supplied class labels and train/validation/test split for the baseline audit.

### 3. Open the notebook

Open [`Retinal_OCT_Image.ipynb`](./Retinal_OCT_Image.ipynb) in Jupyter or Google Colab. Install the libraries used by the notebook and configure its input/output paths for your environment. The paper's experimental stack includes Python, PyTorch/torchvision, NumPy, and scikit-image; additional notebook imports may be required.

### 4. Run or inspect the analysis

The paper's protocol proceeds from duplicate screening and confirmation to cleaning manifests, baseline/retraining experiments, fixed-checkpoint cross-evaluation, and statistical analysis. **Check the notebook's actual cell dependencies and paths before execution**; the repository is a research artifact, not a tested one-command reproduction package.

Experiments in the paper were run using a Colab GPU. Access to the original dataset and sufficient compute/storage are necessary to repeat training; tabular analysis of already-saved results can be less demanding.

### Credentials and security

**Do not commit Kaggle tokens, `kaggle.json`, service-account files, access keys, or other credentials.** Use Colab/Kaggle secrets or environment variables as appropriate. Previously exposed credentials should be revoked and removed from the working tree and Git history; revoking a token alone does not erase it from past commits.

## Interpretation and limitations

- The measured fixed-model change depends on the **specific evaluated checkpoints** and the images removed from the test set; it is not a universal estimate for other models or datasets.
- The audit covers **one public dataset** and **two convolutional backbones**; the deterministic and training-side experiments are confined to ResNet18.
- The release does not provide patient identifiers. The detector cannot establish all possible patient-level overlap, and the **recall of the duplicate-detection protocol was not measured**.
- EfficientNet-B0's seven-seed paired differences depart from normality in the study; nonparametric analyses are therefore relevant when interpreting its paired *t*-test.
- The repository is intended for benchmark integrity research, **not clinical deployment**.

## Paper and citation

**Manuscript:** *Estimating Near-Duplicate Leakage in OCT-C8: Fixed-Model Scoring versus Paired Retraining* (2026).

Publication details (venue, DOI, and the formal bibliographic entry) will be added when available. Until then, reference the manuscript title and repository URL. The final citation and author metadata should follow the approved manuscript.

**Repository:** https://github.com/mhtamim136/oct-c8-near-duplicate-leakage-audit

## License and data use

A repository-wide license has not been specified in the currently visible top-level files. Until the maintainers publish one, do not assume permission to reuse or redistribute the code. The OCT-C8 dataset remains subject to its provider's terms.

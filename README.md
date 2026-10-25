
# Neural Decision Tree Networks with Non-Linear Error-Correcting Nodes

This repository contains the official implementations, experimental notebooks, visualizations, and technical reports for the **Tree-Neuron Network** project series (**ELL409: Machine Learning & Optimization, IIT Delhi**).

Traditional decision trees rely on axis-aligned hyperplanes, limiting their capacity on complex visual manifolds. Standard deep networks update parameters globally via backpropagation, rendering them prone to catastrophic interference on rare edge cases.

This repository presents a **Hybrid Tree-Neuron Architecture** that unifies both paradigms, organized into two sequential research phases:

1. **`basic_network/` (Part I: Foundational Architecture):** Non-linear MLP error-correcting trees, exact 100% training interpolation, empirical double descent, receptive field saliency mapping, and utility-based post-order pruning.
2. **`aug_ensemble_network/` (Part II: Augmentation, Ensembles & Loss Landscapes):** Multi-view stochastic feature banks, triple-diversity random subspace ensembles with hard majority voting, explicit bias-variance decompositions across sample size and epochs, and 1D/2D loss landscape curvature analysis.

---

## Repository Structure

```text
├── basic_network/                     # Phase 1: Foundational Tree-Neuron Network
│   ├── cats_vs_dogs_basic.ipynb       # Binary interpolation, double descent & pruning
│   ├── cifar10_basic.ipynb            # 10-class OvR tree network & logit standardization
│   ├── plots/                         # Double descent curves & saliency maps
│   └── report.pdf               # Technical Report I
│
├── aug_ensemble_network/              # Phase 2: Augmentation, Ensembles & Landscapes
│   ├── cats_vs_dogs_ensemble.ipynb    # Multi-view aug, 9-tree ensemble, Bias-Var & landscapes
│   ├── cifar10_ensemble.ipynb         # 7-member OvR ensemble, dampened scoring & landscapes
│   ├── plots/                         # Bias-Variance sweeps, 1D/2D loss surfaces & saliency
│   └── report.pdf               # Technical Report II
│
└── README.md
```

---

## Core Architectural Paradigm

- Every node is parameterized by a non-linear **Multi-Layer Perceptron (MLP)** operating on continuous embeddings extracted from a shared **ResNet-18** backbone.
- Residual misclassifications are isolated locally using **Jayadeva's error-decomposition (C1, C2, C3, C4)**:
  - **Additive Boost (Child A):** Spawned on Dv \ C2 to rescue false negatives (C3 deficit errors) via analytical offset +w_A.
  - **Subtractive Penalty (Child B):** Spawned on Dv \ C1 to suppress false positives (C4 surplus errors) via analytical offset -w_B.

### System Architecture (Single Tree & Multi-View Ensemble)

```text
                        ┌─────────────────────────────────┐
                        │    Input Image I (128x128x3)    │
                        └─────────────────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼ (Phase 1: Static)                       ▼ (Phase 2: Multi-View)
     ┌───────────────────────────┐               ┌───────────────────────────────────┐
     │ Frozen ResNet-18 Backbone │               │ K-View Stochastic Augmentations   │
     │      x = phi(I) in R^512  │               │   + Frozen ResNet-18 Backbone     │
     └───────────────────────────┘               │     X_aug in R^(K x N x 512)      │
                    │                            └───────────────────────────────────┘
                    ▼                                              │
     ┌───────────────────────────┐                 ┌───────────────┼───────────────┐
     │       Parent Node v       │                 ▼               ▼               ▼
     │  z_v(x) = w2^T ReLU(W1x+b1│           ┌──────────┐    ┌──────────┐    ┌──────────┐
     └───────────────────────────┘           │  Tree 1  │    │  Tree m  │    │  Tree M  │
            /             \                  │ (View 1) │    │ (View k) │    │ (View K) │
  False Neg /               \ False Pos      │ Subspace │    │ Subspace │    │ Subspace │
     (C3)  /                 \  (C4)         └──────────┘    └──────────┘    └──────────┘
          ▼                   ▼                    │               │               │
┌─────────────────┐ ┌─────────────────┐            └───────────────┼───────────────┘
│     Child A     │ │     Child B     │                            ▼
│ (Additive +w_A) │ │ (Penalty -w_B)  │           ┌───────────────────────────────────┐
└─────────────────┘ └─────────────────┘           │        Hard Majority Vote         │
                                                  │   y_hat = mode{y1(x), ..., yM(x)} │
                                                  └───────────────────────────────────┘
```

---

## Key Innovations & Engineering Highlights

### Phase 1: `basic_network/`

- **Exact 100% Training Interpolation:** Achieves exact **100.00%** training accuracy on Kaggle Dogs vs. Cats (N=20,000) using just 34 non-linear nodes, and **98.14%** multiclass training accuracy across 10 One-vs-Rest trees on CIFAR-10 (N=40,000).
- **Empirical Double Descent in Decision Trees:** Demonstrates that neural tree architectures undergo double descent across depth sweeps. Near the interpolation boundary, test error peaks before descending as deep leaf nodes isolate rare outliers without degrading global parent boundaries.
- **Receptive Field Saliency Mapping:** Input-gradient attribution (grad_I z_child) reveals hierarchical division of labor: root nodes capture global silhouettes, while child nodes specialize in localized morphological edges (muzzles, ear contours, wing edges).
- **Utility-Based Post-Order Pruning:** Eliminates sparse leaf branches (tau_min = 5), compressing tree node counts by **21.88%** on Cats vs. Dogs and **13.41%** on CIFAR-10 with zero drop in test accuracy.

### Phase 2: `aug_ensemble_network/`

- **Multi-View Augmented Feature Banks (K-View Extraction):** Overcomes the "single frozen draw" limitation of static backbones by extracting K aligned stochastic augmented views per training image, enforcing transformation invariance across node splits.
- **Triple-Diversity Ensembles with Majority Voting:** Combines distinct augmented views, 85% bootstrap bagging, and Random Feature Subspaces (50%–75% feature projections) across M trees with discrete **Majority Voting**, boosting CIFAR-10 test accuracy by **+14.73%** over Phase 1.
- **Dampened Child-Logit Calibration & sqrt(N_neg / N_pos) Weighting:** Eliminates intermediate depth-2/3 false-positive spikes and prevents overfitted leaf activations from hijacking 10-class OvR argmax scoring.
- **Statistical Bias-Variance Decomposition:** Explicitly tracks Bias^2 and Variance across tree depth (d), sample size (N), and training epochs (E), uncovering a monotonic decay with N and a non-monotonic variance U-turn across epochs E.
- **1D & 2D Loss Landscape Curvature Probing:** Unit-normalized directional perturbations (Delta L(theta* +/- alpha * d)) prove that deep memorizing child nodes sit in sharp, high-curvature canyons, whereas multi-view augmented root nodes converge to wide, flat basins (kappa_child > kappa_clean_root > kappa_aug_root).

---

## Datasets

- **Cats vs. Dogs:** [Kaggle Dogs vs. Cats Redux](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition) (25,000 images, resized to 128 x 128 x 3).
- **CIFAR-10:** [Kaggle CIFAR-10 Dataset](https://www.kaggle.com/c/cifar-10) (50,000 images across 10 classes, upsampled to 64 x 64 in Phase 1 and 128 x 128 in Phase 2).

---

## Experimental Results Summary

### Phase 1: Foundational Single Tree & Pruning (`basic_network/`)

| Benchmark               | Model Regime                   | Total Nodes | Train Acc (%) | Test Acc (%) | Node Compression |
| ----------------------- | ------------------------------ | ----------- | ------------- | ------------ | ---------------- |
| **Cats vs. Dogs**       | Full Interpolated (d=20)       | 32          | **100.00%**   | **94.88%**   | —                |
| **Cats vs. Dogs**       | Post-Order Pruned (tau_min=5)  | **25**      | 99.85%        | **94.86%**   | **-21.88%**      |
| **CIFAR-10 (10-Class)** | Uncalibrated Single OvR (d=12) | 522         | **98.14%**    | **67.90%**   | —                |
| **CIFAR-10 (10-Class)** | Post-Order Pruned (tau_min=5)  | **452**     | 97.80%        | **67.92%**   | **-13.41%**      |

### Phase 2: Multi-View Augmentation & Majority-Vote Ensembles (`aug_ensemble_network/`)

| Benchmark               | Model Configuration                     | Max Depth | Train Acc (%) | Test Acc (%) | Bias^2     | Variance   |
| ----------------------- | --------------------------------------- | --------- | ------------- | ------------ | ---------- | ---------- |
| **Cats vs. Dogs**       | Single Clean Tree (Unaugmented)         | 14        | 99.78%        | 95.42%       | —          | —          |
| **Cats vs. Dogs**       | Single Tree + Multi-View Aug            | 20        | —             | 95.64%       | —          | —          |
| **Cats vs. Dogs**       | **9-Tree Ensemble (Majority Vote)**     | 14        | —             | **95.78%**   | **0.0325** | **0.0196** |
| **CIFAR-10 (10-Class)** | Baseline Clean OvR (128 x 128)          | 8         | 87.67%        | 72.00%       | —          | —          |
| **CIFAR-10 (10-Class)** | Single OvR + Multi-Aug + Dampened       | 4         | —             | 76.35%       | —          | —          |
| **CIFAR-10 (10-Class)** | **7-Tree OvR Ensemble (Majority Vote)** | 8         | —             | **82.63%**   | **0.2312** | **0.0915** |

# Neural Decision Tree Networks with Non-Linear Error-Correcting Nodes

This repository contains the official implementation and technical report for the **Tree-Neuron Network Revisited** project (ELL409: Machine Learning & Optimization, IIT Delhi).

Traditional decision trees rely on axis-aligned hyperplanes, limiting their capacity on complex visual manifolds. Standard deep networks update parameters globally via backpropagation, rendering them prone to catastrophic interference on rare edge cases.

This repository presents a **Hybrid Tree-Neuron Architecture** that unifies both paradigms:

- Every node is parameterized by a non-linear **Multi-Layer Perceptron (MLP)** operating on continuous embeddings extracted from a shared **ResNet-18** backbone.
- Residual misclassifications are isolated locally using **Jayadeva error-decomposition**:
  - **Additive Boost (Child A):** Rescues false negatives (deficit errors).
  - **Subtractive Penalty (Child B):** Suppresses false positives (surplus errors).

## Key Innovations & Engineering Highlights

- **Exact 100% Training Interpolation**  
  Achieves exact **100.00%** training accuracy on Kaggle Dogs vs. Cats (N=20,000) using just 34 non-linear nodes, and **98.14%** multiclass training accuracy across 10 One-vs-Rest trees on CIFAR-10 (N=40,000).

- **Empirical Double Descent in Decision Trees**  
  Demonstrates that modern neural tree architectures undergo double descent across depth sweeps. Near the interpolation boundary, test error peaks before monotonically descending as deep leaf nodes isolate rare outliers without degrading global decision boundaries.

- **Receptive Field Saliency Mapping**  
  Input-gradient attribution (gradient of child activation with respect to input image I) reveals structural division of labor: parent nodes capture global geometric silhouettes, while child nodes specialize in fine-grained anatomical and edge features (e.g., muzzles, ear orientation, wing edges).

- **Utility-Based Post-Order Pruning**  
  Eliminates low-activation leaf branches (tau_min = 5), compressing tree node counts by **21.88%** on Cats vs. Dogs and **13.41%** on CIFAR-10 with zero drop in test accuracy.

- **Z-Score Logit Standardization**  
  Resolves scale mismatches in One-vs-Rest tree ensembles by standardizing unthresholded tree logits prior to argmax arbitration.

## System Architecture

```text
                        ┌─────────────────────────────────┐
                        │   Input Image I (128x128x3)     │
                        └─────────────────────────────────┘
                                         │
                                         ▼
                        ┌─────────────────────────────────┐
                        │   Frozen ResNet-18 Backbone     │
                        │        x = φ(I) ∈ ℝ⁵¹²          │
                        └─────────────────────────────────┘
                                         │
                                         ▼
                        ┌─────────────────────────────────┐
                        │        Parent Node v            │
                        │   z_v(x) = w₂ᵀ ReLU(W₁x + b₁)   │
                        └─────────────────────────────────┘
                                  /             \
                       False Neg /               \ False Pos
                          (C₃)  /                 \  (C₄)
                               ▼                   ▼
     ┌───────────────────────────────────┐       ┌───────────────────────────────────┐
     │      Child A (Additive Boost)     │       │    Child B (Subtractive Penalty)  │
     │    Target: 1 for C₃, 0 otherwise  │       │   Target: 1 for C₄, 0 otherwise   │
     │       Weight Offset: +w_A         │       │       Weight Offset: -w_B         │
     └───────────────────────────────────┘       └───────────────────────────────────┘
```

## Datasets

- **Cats vs. Dogs:** [Kaggle Dogs vs. Cats Redux](https://www.google.com/search?q=https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition) (25,000 images, 80/20 train/test split).
- **CIFAR-10:** [Kaggle CIFAR-10 Dataset](https://www.kaggle.com/c/cifar-10) (50,000 images, 40k/10k train/test split).

## Experimental Results Summary

| Benchmark               | Model Regime      | Total Nodes | Train Acc (%) | Test Acc (%) | Compression |
| ----------------------- | ----------------- | ----------- | ------------- | ------------ | ----------- |
| **Cats vs. Dogs**       | Full Interpolated | 32          | **100.00%**   | **94.88%**   | —           |
| **Cats vs. Dogs**       | Pruned (tau=5)    | **25**      | 99.85%        | **94.86%**   | **21.88%**  |
| **CIFAR-10 (10-Class)** | Full Ensemble     | 522         | **98.14%**    | **67.90%**   | —           |
| **CIFAR-10 (10-Class)** | Pruned Ensemble   | **452**     | 97.80%        | **67.92%**   | **13.41%**  |

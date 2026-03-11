# Neural Network from Scratch + PyTorch Regression Pipeline

Two-part project: a custom neural network library built entirely from scratch using NumPy, and a PyTorch-based regression model for California house price prediction with systematic hyperparameter optimisation.

**Part 1 is built without PyTorch, TensorFlow, or any ML framework** — forward pass, backpropagation, loss computation, and mini-batch SGD are all implemented from first principles.

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3 |
| **Part 1 — From Scratch** | NumPy (matrix operations, backpropagation, gradient computation) |
| **Part 2 — Framework** | PyTorch (autograd, nn.Module), scikit-learn (preprocessing, metrics) |
| **Data Processing** | pandas, NumPy |
| **Visualisation** | Matplotlib |

---

## Part 1: Neural Network Library (NumPy Only)

A complete neural network framework implemented from first principles.

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   CUSTOM NN LIBRARY (NumPy)                      │
│                                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────────┐    │
│  │ Preprocessor │──>│ MultiLayer   │──>│  Loss Layer       │    │
│  │ (min-max     │   │ Network      │   │  MSE / CrossEnt   │    │
│  │  normalise)  │   │              │   │                   │    │
│  └─────────────┘   └──────┬───────┘   └─────────┬─────────┘    │
│                           │                     │                │
│                           v                     v                │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    Layer Stack                              │  │
│  │  Input ──> [Linear + ReLU] ──> [Linear + Sigmoid] ──> Out  │  │
│  │                                                            │  │
│  │  Forward:  z = Wx + b  →  a = activation(z)               │  │
│  │  Backward: ∂L/∂W, ∂L/∂b via chain rule (manual backprop)  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─────────────┐                                                │
│  │   Trainer   │  Mini-batch SGD, configurable epochs,          │
│  │             │  batch size, learning rate, shuffling           │
│  └─────────────┘                                                │
└──────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Implementation Detail |
|---|---|
| LinearLayer | z = Wx + b with Xavier initialisation; stores ∂L/∂W and ∂L/∂b |
| ReLU / Sigmoid | Forward activation + analytical derivative for backward pass |
| MSELossLayer | L = (1/n)Σ(y - ŷ)²; gradient = 2(ŷ - y)/n |
| CrossEntropyLossLayer | Softmax + negative log-likelihood for multi-class classification |
| Preprocessor | Min-max normalisation to [0, 1] fitted on training data |
| Trainer | Mini-batch SGD with epoch-level shuffling and loss tracking |

Tested on the **Iris dataset** (4 features → 3-class classification).

---

## Part 2: House Price Regression (PyTorch)

A production-ready regression pipeline for predicting California median house values from the California Housing dataset (16,512 samples, 9 features).

### Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                  REGRESSION PIPELINE (PyTorch)                   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Preprocessing (scikit-learn ColumnTransformer)          │    │
│  │  Numeric: KNN Imputation → StandardScaler               │    │
│  │  Categorical: Most-Frequent Imputation → OneHotEncoder   │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           v                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Configurable Neural Network (nn.Module)                 │    │
│  │  Pyramid:      256 → 128 → 64 → 32 → 1                 │    │
│  │  Rectangular:  256 → 256 → 256 → 256 → 1                │    │
│  │  Activations:  ReLU / Sigmoid / LeakyReLU                │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           v                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Training: Adam + Early Stopping + L2 Regularisation     │    │
│  │  Evaluation: RMSE, MAE, R²                               │    │
│  │  Hyperparameter Search: Grid (2,016 combos × 10-fold CV) │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### Best Configuration (from Grid Search)

| Parameter | Value |
|---|---|
| Learning rate | 0.01 |
| Hidden layers | 4 |
| Neurons per layer | 256 |
| Architecture | Rectangular |
| Activation | ReLU |
| Batch size | 32 |
| Weight decay | 0.0 |

### Hyperparameter Search Space

| Parameter | Values Tested |
|---|---|
| Learning rate | 0.001, 0.01 |
| Weight decay | 0.0, 0.001 |
| Hidden layers | 2, 4, 6 |
| Neurons | 32, 64, 128, 256 |
| Batch size | 32, 64, 128 |
| Architecture | Pyramid, Rectangular |
| Activation | ReLU, Sigmoid, LeakyReLU |

Total: **2,016 combinations** evaluated with 10-fold cross-validation.

---

## Project Structure

```
├── part1_nn_lib.py                 # Custom neural network library (NumPy only)
├── part2_house_value_regression.py  # House price regression pipeline (PyTorch)
├── housing.csv                      # California housing dataset (16,512 samples)
├── iris.dat                         # Iris dataset for Part 1 testing
├── requirements.txt
└── README.md
```

---

## Getting Started

```bash
git clone https://github.com/jiayihuang1/neural-network-from-scratch.git
cd neural-network-from-scratch

pip install -r requirements.txt

# Part 1 — run the custom NN library on Iris
python part1_nn_lib.py

# Part 2 — train and evaluate the house price regressor
python part2_house_value_regression.py
```

---

## Academic Context

Developed as coursework for **COMP70050 Introduction to Machine Learning** at Imperial College London (MSc AI).

## Contributors

- Ethan Chia Wei Fong
- Benjamin Ang
- Catalina Tan
- Jia Yi Huang

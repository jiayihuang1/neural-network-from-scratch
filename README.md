# Neural Network from Scratch

Two-part project: a custom neural network library built from scratch using only NumPy, and a PyTorch-based regression model for California house price prediction with hyperparameter optimisation.

## Part 1: Neural Network Library (NumPy)

A fully custom neural network implementation with no framework dependencies:

- **Layers**: Linear (dense), ReLU, Sigmoid
- **Loss functions**: MSE (regression), Cross-Entropy (classification)
- **Training**: Mini-batch SGD with Xavier weight initialisation
- **Preprocessing**: Min-max normalisation
- Tested on the Iris dataset (3-class classification)

## Part 2: House Price Regression (PyTorch)

A production-ready regression pipeline for predicting California median house values:

- **Model**: Configurable multi-layer neural network (supports pyramid and rectangular architectures)
- **Preprocessing**: KNN imputation, StandardScaler (numeric), OneHotEncoder (categorical) via scikit-learn
- **Training**: Adam optimiser with early stopping, L2 regularisation, batch training
- **Hyperparameter search**: Grid search over 2,016 combinations with 10-fold CV
- **Evaluation**: RMSE, MAE, R² score with visualisations

### Best Configuration Found

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.01 |
| Hidden layers | 4 |
| Neurons | 256 |
| Architecture | Rectangular |
| Activation | ReLU |
| Batch size | 32 |

## Project Structure

```
├── part1_nn_lib.py                 # Custom neural network library (NumPy only)
├── part2_house_value_regression.py  # House price regression (PyTorch)
├── housing.csv                      # California housing dataset (16,512 samples)
├── iris.dat                         # Iris dataset for Part 1 testing
├── requirements.txt
└── README.md
```

## Tech Stack

**Part 1**: Python, NumPy

**Part 2**: Python, PyTorch, scikit-learn, pandas, matplotlib

## Usage

```bash
pip install -r requirements.txt

# Part 1 — run the neural net library example
python part1_nn_lib.py

# Part 2 — train and evaluate the regressor
python part2_house_value_regression.py
```

## Academic Context

Developed as coursework for **COMP70050 Introduction to Machine Learning** at Imperial College London (MSc AI).

## Contributors

- Ethan Chia Wei Fong
- Benjamin Ang
- Catalina Tan
- Jia Yi Huang

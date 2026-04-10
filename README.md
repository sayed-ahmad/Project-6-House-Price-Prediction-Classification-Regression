# Project 6 — House Price Prediction (Classification & Regression)

This project implements a complete **Machine Learning workflow** to analyse and predict house prices. It covers both major types of supervised learning:

| Task | Goal |
|------|------|
| **Classification** | Predict whether a house is *"Expensive"* or *"Not Expensive"* |
| **Regression** | Predict the exact house price in US dollars |

---

## Project Structure

```
.
├── house_price_prediction.py   # Main ML workflow script
├── requirements.txt            # Python dependencies
├── data/
│   └── house_prices.csv        # Dataset (auto-generated on first run)
└── plots/                      # Output visualisations
    ├── 01_price_distribution.png
    ├── 02_correlation_heatmap.png
    ├── 03_price_by_neighborhood.png
    ├── 04_sqft_vs_price.png
    ├── 05_confusion_matrix.png
    ├── 06_classifier_comparison.png
    ├── 07_clf_feature_importance.png
    ├── 08_actual_vs_predicted.png
    ├── 09_regressor_comparison.png
    └── 10_reg_feature_importance.png
```

---

## Dataset

A realistic **synthetic dataset** of 1 000 houses is generated on the first run and saved to `data/house_prices.csv`. Each row represents a house with the following features:

| Feature | Description |
|---------|-------------|
| `sqft_living` | Interior living area (sq ft) |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `lot_size` | Total lot area (sq ft) |
| `floors` | Number of floors |
| `garage_spaces` | Number of garage spaces |
| `year_built` | Year the house was built |
| `age` | Age of the house (years) |
| `condition` | Overall condition (1–5) |
| `view_quality` | View quality score (0–4) |
| `neighborhood` | Neighbourhood type |
| `house_type` | Type of house |
| `price` | Sale price in USD (target) |

---

## ML Workflow

### 1. Exploratory Data Analysis (EDA)
- Summary statistics and missing-value check
- Price distribution (raw and log-transformed)
- Feature correlation heatmap
- Price by neighbourhood (box plots)
- Living area vs price scatter

### 2. Feature Engineering & Pre-processing
- Label encoding of categorical features (`neighborhood`, `house_type`)
- Standard scaling for linear models
- Classification label: houses at or above the **median price** are labelled *"Expensive"*

### 3. Classification Task
Four models are trained and compared using accuracy and 5-fold cross-validation:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

The best model's confusion matrix and a per-model accuracy bar chart are saved.

### 4. Regression Task
Four models are trained and compared using MAE, RMSE, and R²:
- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Actual-vs-predicted scatter, residual plot, and model comparison charts are saved.

---

## Getting Started

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the workflow

```bash
python house_price_prediction.py
```

All results are printed to the console and all plots are saved to the `plots/` directory.

---

## Sample Results

```
CLASSIFICATION TASK
  Gradient Boosting — Accuracy: 0.9750

REGRESSION TASK
  Gradient Boosting — R²: 0.9897 | MAE: $29,527 | RMSE: $36,282
```
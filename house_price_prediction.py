"""
House Price Prediction - Classification & Regression
=====================================================
This script implements a complete Machine Learning workflow for house price prediction.

Two tasks are performed:
1. Classification: Predict whether a house is "Expensive" or "Not Expensive"
2. Regression:     Predict the exact house price in dollars
"""

import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
RANDOM_STATE = 42
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Price threshold (median) used to define "Expensive" vs "Not Expensive"
EXPENSIVE_LABEL = "Expensive"
NOT_EXPENSIVE_LABEL = "Not Expensive"


# ─────────────────────────────────────────────
# 1. Data Generation / Loading
# ─────────────────────────────────────────────

def generate_dataset(n_samples: int = 1000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a realistic synthetic house-price dataset."""
    rng = np.random.default_rng(random_state)

    neighborhoods = ["Downtown", "Suburbs", "Rural", "Urban", "Waterfront"]
    neighborhood_premium = {
        "Downtown": 50_000,
        "Suburbs": 20_000,
        "Rural": -30_000,
        "Urban": 30_000,
        "Waterfront": 100_000,
    }
    house_types = ["Single Family", "Condo", "Townhouse", "Multi-Family"]

    # Core features
    sqft_living = rng.integers(500, 5001, n_samples, dtype=np.int32)
    bedrooms = rng.integers(1, 8, n_samples, dtype=np.int32)
    bathrooms = np.clip(rng.normal(bedrooms * 0.7, 0.5, n_samples), 1, 8).round(1)
    lot_size = rng.integers(1000, 20001, n_samples, dtype=np.int32)
    year_built = rng.integers(1950, 2024, n_samples, dtype=np.int32)
    floors = rng.choice([1, 1.5, 2, 2.5, 3], n_samples)
    garage = rng.integers(0, 4, n_samples, dtype=np.int32)
    neighborhood = rng.choice(neighborhoods, n_samples)
    house_type = rng.choice(house_types, n_samples)
    condition = rng.integers(1, 6, n_samples, dtype=np.int32)   # 1–5
    view_quality = rng.integers(0, 5, n_samples, dtype=np.int32)  # 0–4

    # Derived / engineered features
    age = 2024 - year_built
    price_per_sqft_base = 150 + condition * 20 + view_quality * 15

    # Base price model
    base_price = (
        sqft_living * price_per_sqft_base
        + bedrooms * 5_000
        + bathrooms * 8_000
        + garage * 15_000
        + lot_size * 2
        - age * 500
        + np.array([neighborhood_premium[n] for n in neighborhood])
        + rng.normal(0, 20_000, n_samples)
    )
    price = np.clip(base_price, 50_000, 3_000_000).round(-2)

    df = pd.DataFrame({
        "sqft_living": sqft_living,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "lot_size": lot_size,
        "floors": floors,
        "garage_spaces": garage,
        "year_built": year_built,
        "age": age,
        "condition": condition,
        "view_quality": view_quality,
        "neighborhood": neighborhood,
        "house_type": house_type,
        "price": price,
    })
    return df


def load_data(n_samples: int = 1000) -> pd.DataFrame:
    """Load the dataset, generating it if no CSV file is found."""
    csv_path = os.path.join("data", "house_prices.csv")
    if os.path.exists(csv_path):
        print(f"[INFO] Loading dataset from '{csv_path}'")
        df = pd.read_csv(csv_path)
    else:
        print("[INFO] Generating synthetic dataset …")
        df = generate_dataset(n_samples)
        os.makedirs("data", exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Dataset saved to '{csv_path}'")
    return df


# ─────────────────────────────────────────────
# 2. Exploratory Data Analysis
# ─────────────────────────────────────────────

def exploratory_data_analysis(df: pd.DataFrame) -> None:
    """Print summary statistics and save EDA plots."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"\nDataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nBasic statistics:")
    print(df.describe())
    print("\nMissing values:")
    print(df.isnull().sum())
    print(f"\nPrice range: ${df['price'].min():,.0f} – ${df['price'].max():,.0f}")
    print(f"Median price: ${df['price'].median():,.0f}")

    # --- Price distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["price"], bins=40, color="steelblue", edgecolor="white")
    axes[0].set_title("Price Distribution")
    axes[0].set_xlabel("Price ($)")
    axes[0].set_ylabel("Count")
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))

    axes[1].hist(np.log1p(df["price"]), bins=40, color="darkorange", edgecolor="white")
    axes[1].set_title("Log-Price Distribution")
    axes[1].set_xlabel("log(Price + 1)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_price_distribution.png"), dpi=120)
    plt.close()

    # --- Correlation heat-map (numeric columns only) ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    plt.figure(figsize=(12, 8))
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_correlation_heatmap.png"), dpi=120)
    plt.close()

    # --- Price by neighbourhood ---
    plt.figure(figsize=(10, 5))
    order = df.groupby("neighborhood")["price"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="neighborhood", y="price", order=order, palette="Set2")
    plt.title("Price Distribution by Neighborhood")
    plt.xlabel("Neighborhood")
    plt.ylabel("Price ($)")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_price_by_neighborhood.png"), dpi=120)
    plt.close()

    # --- Scatter: sqft_living vs price ---
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(df["sqft_living"], df["price"],
                          c=df["condition"], cmap="viridis", alpha=0.5, s=15)
    plt.colorbar(scatter, label="Condition")
    plt.title("Living Area vs Price (coloured by Condition)")
    plt.xlabel("Living Area (sqft)")
    plt.ylabel("Price ($)")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_sqft_vs_price.png"), dpi=120)
    plt.close()

    print(f"\n[INFO] EDA plots saved to '{PLOTS_DIR}/'")


# ─────────────────────────────────────────────
# 3. Feature Engineering & Pre-processing
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame, threshold: float | None = None):
    """
    Encode categoricals, create the classification label, and split into
    features / targets.

    Returns
    -------
    X            : feature DataFrame (numeric, encoded)
    y_reg        : continuous price target
    y_clf        : binary classification label (0 = Not Expensive, 1 = Expensive)
    threshold    : price threshold used for the classification label
    feature_names: list of feature column names
    """
    df = df.copy()

    # Encode categorical columns with separate encoders
    le_neighborhood = LabelEncoder()
    le_house_type = LabelEncoder()
    df["neighborhood_enc"] = le_neighborhood.fit_transform(df["neighborhood"])
    df["house_type_enc"] = le_house_type.fit_transform(df["house_type"])

    # Feature matrix
    feature_cols = [
        "sqft_living", "bedrooms", "bathrooms", "lot_size",
        "floors", "garage_spaces", "year_built", "age",
        "condition", "view_quality",
        "neighborhood_enc", "house_type_enc",
    ]
    X = df[feature_cols].copy()
    y_reg = df["price"].copy()

    # Classification label: above / below median price
    if threshold is None:
        threshold = float(y_reg.median())
    y_clf = (y_reg >= threshold).astype(int)

    print(f"\n[INFO] Classification threshold (median price): ${threshold:,.0f}")
    print(f"       Not Expensive (0): {(y_clf == 0).sum():,} samples "
          f"({(y_clf == 0).mean():.1%})")
    print(f"       Expensive     (1): {(y_clf == 1).sum():,} samples "
          f"({(y_clf == 1).mean():.1%})")

    return X, y_reg, y_clf, threshold, feature_cols


# ─────────────────────────────────────────────
# 4. Classification Task
# ─────────────────────────────────────────────

def run_classification(X: pd.DataFrame, y_clf: pd.Series,
                       feature_names: list) -> None:
    """Train and evaluate multiple classifiers."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION TASK")
    print(f"  Goal: Predict '{EXPENSIVE_LABEL}' vs '{NOT_EXPENSIVE_LABEL}'")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
    )

    # Wrap Logistic Regression in a Pipeline so scaling is applied consistently
    # during both fit/predict and cross-validation.
    classifiers = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }

    results = {}
    best_name, best_acc = "", 0.0

    print(f"\n{'Model':<25} {'Accuracy':>10} {'CV Accuracy':>13}")
    print("-" * 50)

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        cv_scores = cross_val_score(clf, X, y_clf, cv=5, scoring="accuracy")
        cv_mean = cv_scores.mean()

        results[name] = {"clf": clf, "y_pred": y_pred, "acc": acc, "cv": cv_mean}
        print(f"{name:<25} {acc:>10.4f} {cv_mean:>13.4f}")

        if acc > best_acc:
            best_acc, best_name = acc, name

    print(f"\n[INFO] Best classifier: {best_name} (accuracy = {best_acc:.4f})")

    # Detailed report for best model
    best = results[best_name]
    print(f"\nClassification Report — {best_name}:")
    print(classification_report(y_test, best["y_pred"],
                                target_names=[NOT_EXPENSIVE_LABEL, EXPENSIVE_LABEL]))

    # --- Confusion matrix ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, best["y_pred"],
        display_labels=[NOT_EXPENSIVE_LABEL, EXPENSIVE_LABEL],
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "05_confusion_matrix.png"), dpi=120)
    plt.close()

    # --- Classifier accuracy comparison bar chart ---
    names = list(results.keys())
    accs = [results[n]["acc"] for n in names]
    cvs = [results[n]["cv"] for n in names]
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, accs, width, label="Test Accuracy", color="steelblue")
    ax.bar(x + width / 2, cvs, width, label="CV Accuracy (5-fold)", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Classifier Comparison")
    ax.legend()
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "06_classifier_comparison.png"), dpi=120)
    plt.close()

    # --- Feature importance (best tree-based model) ---
    best_clf = best["clf"]
    # Unwrap Pipeline if necessary
    estimator = best_clf.named_steps["clf"] if isinstance(best_clf, Pipeline) else best_clf
    if hasattr(estimator, "feature_importances_"):
        importances = pd.Series(estimator.feature_importances_, index=feature_names)
        importances = importances.sort_values(ascending=True)

        plt.figure(figsize=(8, 5))
        importances.plot(kind="barh", color="steelblue")
        plt.title(f"Feature Importances — {best_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "07_clf_feature_importance.png"), dpi=120)
        plt.close()


# ─────────────────────────────────────────────
# 5. Regression Task
# ─────────────────────────────────────────────

def run_regression(X: pd.DataFrame, y_reg: pd.Series,
                   feature_names: list) -> None:
    """Train and evaluate multiple regression models."""
    print("\n" + "=" * 60)
    print("REGRESSION TASK")
    print("  Goal: Predict the exact house price in dollars")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=RANDOM_STATE
    )

    # Wrap Linear Regression in a Pipeline so scaling is applied consistently.
    regressors = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ]),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
    }

    results = {}
    best_name, best_r2 = "", -np.inf

    print(f"\n{'Model':<25} {'MAE':>15} {'RMSE':>15} {'R²':>10}")
    print("-" * 67)

    for name, reg in regressors.items():
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {"reg": reg, "y_pred": y_pred, "mae": mae,
                         "rmse": rmse, "r2": r2}
        print(f"{name:<25} {mae:>15,.0f} {rmse:>15,.0f} {r2:>10.4f}")

        if r2 > best_r2:
            best_r2, best_name = r2, name

    best = results[best_name]
    print(f"\n[INFO] Best regressor: {best_name} (R² = {best_r2:.4f})")

    # --- Actual vs Predicted scatter ---
    y_pred_best = best["y_pred"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_test, y_pred_best, alpha=0.4, s=15, color="steelblue")
    lims = [min(y_test.min(), y_pred_best.min()),
            max(y_test.max(), y_pred_best.max())]
    axes[0].plot(lims, lims, "r--", linewidth=1)
    axes[0].set_xlabel("Actual Price ($)")
    axes[0].set_ylabel("Predicted Price ($)")
    axes[0].set_title(f"Actual vs Predicted — {best_name}")
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))

    residuals = y_test.values - y_pred_best
    axes[1].scatter(y_pred_best, residuals, alpha=0.4, s=15, color="darkorange")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Predicted Price ($)")
    axes[1].set_ylabel("Residual ($)")
    axes[1].set_title(f"Residual Plot — {best_name}")
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "08_actual_vs_predicted.png"), dpi=120)
    plt.close()

    # --- Regression model comparison ---
    names = list(results.keys())
    r2_vals = [results[n]["r2"] for n in names]
    mae_vals = [results[n]["mae"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["steelblue", "darkorange", "green", "red"]

    axes[0].bar(names, r2_vals, color=colors)
    axes[0].set_title("R² Score Comparison")
    axes[0].set_ylabel("R²")
    axes[0].set_ylim(0, 1.1)
    axes[0].tick_params(axis="x", rotation=10)
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)

    axes[1].bar(names, mae_vals, color=colors)
    axes[1].set_title("Mean Absolute Error Comparison")
    axes[1].set_ylabel("MAE ($)")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    axes[1].tick_params(axis="x", rotation=10)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "09_regressor_comparison.png"), dpi=120)
    plt.close()

    # --- Feature importance (best tree-based regressor) ---
    best_reg = best["reg"]
    # Unwrap Pipeline if necessary
    estimator = best_reg.named_steps["reg"] if isinstance(best_reg, Pipeline) else best_reg
    if hasattr(estimator, "feature_importances_"):
        importances = pd.Series(estimator.feature_importances_, index=feature_names)
        importances = importances.sort_values(ascending=True)

        plt.figure(figsize=(8, 5))
        importances.plot(kind="barh", color="darkorange")
        plt.title(f"Feature Importances — {best_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "10_reg_feature_importance.png"), dpi=120)
        plt.close()


# ─────────────────────────────────────────────
# 6. Main entry point
# ─────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("House Price Prediction — ML Workflow")
    print("=" * 60)

    # Load / generate dataset
    df = load_data()

    # EDA
    exploratory_data_analysis(df)

    # Pre-processing
    X, y_reg, y_clf, threshold, feature_names = preprocess(df)

    # Classification
    run_classification(X, y_clf, feature_names)

    # Regression
    run_regression(X, y_reg, feature_names)

    print("\n" + "=" * 60)
    print(f"All plots saved to '{PLOTS_DIR}/'")
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

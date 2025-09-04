# 🏠 House Price Prediction Intervals

[**Kaggle Competition: Prediction Interval Competition II - House
Price**](https://www.kaggle.com/competitions/prediction-interval-competition-ii-house-price)

------------------------------------------------------------------------

## 📌 Purpose

The goal of this project is to **create a regression model for house
sale prices** that generates the **narrowest possible prediction
intervals** while ensuring reliable coverage.
- Target coverage: **90% nominal marginal coverage** (α = 0.1)
- Metric: **Mean Winkler Interval Score**
- Motivation: Unlike point predictions, prediction intervals quantify
**uncertainty**, making the model more useful in practical
decision-making.

------------------------------------------------------------------------

## 📂 Data

-   **Dataset:** Provided by the Kaggle competition.

-   **Target variable:** House sale price.

-   **Features:** A mix of numerical and categorical attributes
    describing house characteristics.

-   **Splitting strategy:**

    ``` python
    # Initial split into training and testing
    X_train, X_test, y_sale_price_train, y_sale_price_test = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42)

    # Further split test into test and validation
    X_test, X_val, y_sale_price_test, y_sale_price_val = train_test_split(
        X_test, y_sale_price_test, test_size=0.5, random_state=42)

    # Split training into train and calibration sets
    X_train, X_calib, y_sale_price_train, y_calib = train_test_split(
        X_train, y_sale_price_train, test_size=0.2, random_state=42)
    ```

------------------------------------------------------------------------

## 🔄 Workflow

### Stepwise Process

1.  **Data Loading** -- Import training and test datasets from Kaggle.
2.  **Preprocessing & Feature Engineering** -- Handle missing values,
    extract date features, engineer house-level features, encode
    categoricals.
3.  **Data Splitting** -- Create train/validation/calibration/test
    splits.
4.  **Model Training** -- Train base regressors (linear, tree-based,
    boosting).
5.  **Ensemble Learning** -- Combine predictions for improved
    robustness.
6.  **Prediction Intervals** -- Apply conformal prediction to calibrate
    intervals.
7.  **Evaluation** -- Use Winkler Interval Score + empirical coverage.
8.  **Submission** -- Export CSV for Kaggle leaderboard.

------------------------------------------------------------------------

### Visual Workflow

``` mermaid
flowchart TD
    A[📥 Data Loading<br/>Import CSVs] --> B[🛠 Preprocessing<br/>Handle missing values, encode categoricals, feature engineering]
    B --> C[🔀 Data Splitting<br/>Train, Validation, Calibration, Test]
    C --> D[🤖 Base Models<br/>Tree-based, Boosting]
    D --> E[🧩 Ensemble Model<br/>Stacked predictions]
    E --> F[📏 Conformal Prediction<br/>Calibrate intervals]
    F --> G[📊 Evaluation<br/>Winkler Score + Coverage]
    G --> H[📤 Kaggle Submission<br/>Generate submission.csv]
```

------------------------------------------------------------------------

## ⚙️ Methodology

-   **Base Models:** LightGBM, XGBoost, CatBoost, HistGradientBoosting.
-   **Ensemble:** StackingRegressor combining all four models for both
    lower (α = 0.05) and upper (α = 0.95) quantiles.
-   **Conformal Prediction:** Residual-based calibration to ensure 90%
    coverage.
-   **Hyperparameter Tuning:** Random search for sharper
    intervals.
-   **Final Predictions:** Ensemble + conformal calibration outputs
    point + interval predictions.

------------------------------------------------------------------------

## 📏 Evaluation

The competition evaluates models using the **Winkler Interval Score**:

$$
W_\alpha =
\begin{cases}
(u - l) + \frac{2}{\alpha}(l - y), & \text{if } y < l \\
(u - l), & \text{if } l \leq y \leq u \\
(u - l) + \frac{2}{\alpha}(y - u), & \text{if } y > u
\end{cases}
$$

Where:

- \( y \): true sale price  
- \( u \): upper prediction interval  
- \( l \): lower prediction interval
-  l : lower prediction interval 
- \( α \): miscoverage rate (0.1 for 90% coverage)

👉 **Interpretation:** Lower Winkler scores = tighter intervals with
correct coverage.

------------------------------------------------------------------------

## 🚀 Usage

### 1️⃣ Requirements

Install dependencies (recommend using a virtual environment):

``` bash
pip install -r requirements.txt
```

Example `requirements.txt`:

    numpy
    pandas
    scikit-learn
    matplotlib
    seaborn
    xgboost
    lightgbm
    catboost

### 2️⃣ Run Notebook

-   Open the notebook:

    ``` bash
    jupyter notebook ensemble-model-5-interval-prediction.ipynb
    ```

-   Execute cells sequentially to:

    1.  Load and preprocess data
    2.  Train base models and ensemble
    3.  Apply conformal calibration
    4.  Evaluate with Winkler score
    5.  Generate submission for Kaggle

### 3️⃣ Kaggle Submission

-   Upload the generated submission file (`submission.csv`) to the
    [Kaggle competition
    page](https://www.kaggle.com/competitions/prediction-interval-competition-ii-house-price).

------------------------------------------------------------------------

## 📊 Results

-   **Ensemble model with conformal prediction** outperformed individual
    models.
-   **Benchmark Performance:**
    -   Minimum Winkler Score (train): **332,124.3945**
    -   Mean Winkler Score (test): **332,124.45**
    -   Empirical Coverage: **91.6%** (close to target 90%)
-   **Kaggle Leaderboard:** Ranked **138th out of 2,198 entrants (~top
    6%)** 🎉

------------------------------------------------------------------------

## ✅ Conclusion

This project successfully demonstrated how **ensemble learning combined
with conformal prediction** can generate **narrow yet reliable
prediction intervals** for house sale prices. By targeting **90% nominal
coverage**, the model achieved an empirical coverage of **91.6%**,
validating the robustness of the approach.

**Key takeaways:**
- Ensemble models boost predictive power over single regressors.
- Conformal prediction ensures calibrated and trustworthy intervals.
- Achieved **top 6% ranking** in a competitive Kaggle setting.

**Future Directions:**
- Advanced ensembles: LightGBM, CatBoost, and tuned XGBoost.
- Explore Bayesian approaches for probabilistic intervals.
- Domain-specific feature engineering to further reduce Winkler scores.

------------------------------------------------------------------------

📌 **Author:** Priyansh Keshari
📌 **Competition:** [Prediction Interval Competition II - House
Price](https://www.kaggle.com/competitions/prediction-interval-competition-ii-house-price)

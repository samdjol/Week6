# Predicting the Size of Forest Fires

This repository contains a Jupyter Notebook that performs an in-depth exploratory analysis and modeling process to predict the burned area of forest fires. The project uses the "Forest Fires" dataset from the UCI Machine Learning Repository.

The analysis follows a complete data science workflow, from data preprocessing and feature engineering to model building and evaluation. It concludes that predicting the exact fire size is challenging with linear models, and pivots to a classification approach to predict whether a fire will be "large" or "small".

---

## Dataset

The dataset was sourced from the UCI Machine Learning Repository using the `ucimlrepo` library.

* **ID:** 162
* **Name:** Forest Fires
* **Target Variable:** `area` - The total burned area of the forest (in ha).
* **Features:** The notebook utilizes 12 features, including:
    * Spatial coordinates (`X`, `Y`)
    * Temporal data (`month`, `day`)
    * Fire Weather Index (FWI) components (`FFMC`, `DMC`, `DC`, `ISI`)
    * Meteorological conditions (`temp`, `RH`, `wind`, `rain`)

---

## Methodology and Workflow

The notebook follows a detailed, iterative process to find the best possible model.

### 1. Data Exploration and Preprocessing
* The dataset is loaded and its basic properties are examined.
* The target variable, `area`, is found to be heavily right-skewed, with a large number of zero-value entries. To handle this for regression, a **log transformation** (`np.log1p`) is applied.

### 2. Feature Engineering
* **Categorical Encoding:** The `month` and `day` columns are initially one-hot encoded.
* **Cyclical Feature Transformation:** To better capture the seasonal nature of wildfires, the `month` feature is transformed into cyclical sine and cosine components (`month_sin`, `month_cos`). This provides a continuous representation of seasonality. The `day` feature is ultimately dropped as it shows little correlation with the target.
* **Power & Log Transformations:** Based on Box-Cox analysis, several transformations are tested on predictors like `FFMC` and `RH` to improve linearity and model fit. This includes trying log, square root, and inverse transformations.

### 3. Regression Modeling
Multiple linear regression models were built and evaluated to predict the log-transformed `area`.

* **Initial OLS Model:** The first model showed an extremely low R-squared value, indicating a poor fit.
* **Multicollinearity Analysis:** A **Variance Inflation Factor (VIF)** check and correlation heatmap revealed high multicollinearity, particularly between `DC` (Drought Code) and `month_sin`.
* **Model Refinement:** Features with high VIF scores were removed, and different combinations of transformed features were tested.
* **Interaction Terms:** To capture more complex relationships, interaction terms (e.g., `temp * month_cos`) were introduced.
* **Regularization:** **Ridge** and **Lasso** regression models were also tested but showed no significant improvement.

**Finding:** Despite extensive feature engineering and refinement, all regression models failed to produce a meaningful R-squared value, suggesting that a simple linear relationship is not sufficient to predict fire area.

### 4. Classification Modeling
Given the poor performance of regression, the problem was reframed as a binary classification task: predicting whether a fire will be "large" or "small".

* **Thresholding:** A binary target (`target_bi`) was created. Two thresholds were tested:
    1.  **75th Percentile:** Classified fires larger than ~6.57 ha as "large" (Class 1). This resulted in a model with high accuracy but very poor recall for the minority class.
    2.  **50th Percentile (Median):** Classified fires larger than ~0.52 ha as "large". This created a more balanced dataset and led to more robust models.
* **Logistic Regression:** A Logistic Regression model was trained on the final set of engineered features.

---

## Results and Conclusion

The final and most effective model was a **Logistic Regression classifier** using the median fire size as the threshold.

* **Best Features:** The model performed best with a selection of original and transformed features, including `temp`, `DMC`, `sqrt_FFMC`, and `logRH`. Features with high multicollinearity (`DC`, `month_sin`) were removed.
* **Performance:** The final model achieved an **accuracy of approximately 58%**. While this is a modest result, it demonstrates a balanced performance across precision and recall for both classes, making it more reliable than the skewed 75th-percentile model.

**Final Thoughts:**
Predicting the precise area of a forest fire with this dataset is extremely difficult using linear regression. The relationships between weather conditions and fire size are likely highly non-linear. A classification approach is more practical and yields a model that, while not perfect, provides a balanced predictive capability. For future work, more complex, non-linear models (e.g., Random Forest, Gradient Boosting, or neural networks) would be the logical next step to potentially capture the intricate dynamics of wildfire spread.

---

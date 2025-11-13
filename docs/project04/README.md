# Project 4: Predicting Titanic Fare Using Regression
> Exploring which features drive Titanic ticket prices and which regression models capture fare patterns best.

## Project Overview
In this project, we dig into what drives Titanic ticket prices. Using features like age, sex, class, and family size, we test multiple regression models to see which ones capture fare patterns best and which factors actually matter. The goal is to understand how features influence fare, evaluate different regression techniques, and highlight challenges like skewed data and outliers.

---

## Key Steps in This Project

## 1. Import and Inspect the Data
- Loaded the Titanic dataset directly from the seaborn library.
- Inspected the dataset using `.info()`, `.head()`, `.describe()`, and `.isnull().sum()`.
- Explored correlations among numeric features.

  - **Insights:**
    - 891 data instances and 15 features.
    - Missing values in `age`, `deck`, and `embark_town`.
    - Several categorical and boolean features require numeric encoding for modeling.

---

## 2. Data Exploration and Preparation

### 2.1 Handle Missing Values
- Imputed missing `age` values with the median.
- Dropped rows with missing `fare` values.

### 2.2 Feature Engineering
- Added `family_size` = `sibsp` + `parch` + 1.
- Converted categorical features (`sex`, `embarked`, `class`, etc.) into numeric/dummy variables.

  - **Rationale:**
    - Family size and demographics likely influence fare.
    - Numeric encoding ensures models can process categorical inputs.

---

## 3. Feature Selection and Target Definition
- Input features explored for modeling:  
   - **Case 1:** - `age`
   - **Case 2:** - `family_size`
   - **Case 3:** - `age` + `family_size`
   - **Case 4:** - `age`, `family_size`, `sex`, and `pclass`

- Target variable: `fare`.

**Reasoning:**  
- Comparing feature combinations helps identify which factors are most important for predicting fares.

---

## 4. Data Splitting and Stratification
- Split into training and test sets using train_test_split().
- Ensured splits were consistent across all four feature cases for fair comparison.
- 
---

## 5. Model Training and Evaluation

### 5.1 Linear Regression
- Trained linear regression models for all four feature cases.
- Evaluated using R², RMSE, and MAE.
- Observed that Case 4 (age + family_size + sex + pclass) had the best predictive power.

### 5.2 Regularized Models
- Ridge Regression (L2): Reduced overfitting, but small improvement over linear regression.
- ElasticNet (L1 + L2): Balanced regularization and feature selection, performed slightly better than Ridge.

### 5.3 Polynomial Regression
- Degree 3 (cubic): Captured non-linear trends, provided the best performance overall.
- Degree 5: Overfit the data, R² dropped significantly despite visually similar curves.

### Model Performance Summary
| Model                         | R²    | RMSE  | MAE   |
| ----------------------------- | ----- | ----- | ----- |
| Linear Regression             | 0.399 | 29.49 | 20.08 |
| Ridge Regression              | 0.400 | 29.47 | 20.05 |
| ElasticNet                    | 0.429 | 28.75 | 17.39 |
| Polynomial Regression (deg 3) | 0.506 | 26.72 | 15.05 |
| Polynomial Regression (deg 5) | 0.096 | 36.16 | 18.24 |

---

## 6. Final Thoughts & Insights

### Model Comparison Summary
- Most useful features: Passenger class (pclass) was the strongest predictor, followed by family size; age had a minor effect.
- Best performing model: Polynomial regression (degree 3).
- Complexity & regularization: Some complexity (degree 3) improved performance, too much (degree 5) caused overfitting. Regularization slightly stabilized results but didn’t outperform the cubic polynomial.

**Challenges:** Fare prediction is tricky due to skewed values, outliers, and multiple interacting features.

**Optional next steps:**
- Explore additional features like deck, embarked, or alone.
- Flip the target to predict age instead of fare.
- Apply log transformation to fare to address skew and improve regression performance.

---

## How to Run

1. **Open the Project Notebook**  
   Navigate to `notebooks/project04` and open the Jupyter notebook:  
   `ml04_beaderstadt.ipynb`

2. **Select the Correct Kernel**  
   Ensure the notebook uses the correct Python environment where required libraries are installed.

3. **Clear Kernel / Outputs (Optional)**  
   Use Kernel -> Restart & Clear Outputs to start fresh and avoid stale variables or plots.

4. **Run the Notebook**  
   Execute cells sequentially to load data, prepare features, train models, and visualize results.

5. **View Results**  
   Review outputs such as R², RMSE, MAE, and plots to analyze model performance and feature impact.

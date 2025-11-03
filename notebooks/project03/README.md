# Project 3: Beaderstadt Titanic Classification
> Building and evaluating classifiers to predict survival on the Titanic dataset.

## Project Overview
In this project, I extend my previous exploration of the Titanic dataset to build and evaluate machine learning classifiers. The goal is to predict passenger survival using three models: Decision Tree, Support Vector Machine (SVM), and Neural Network. I explore multiple feature sets, train models, evaluate performance, and visualize the results to understand which factors best predict survival.

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
- Filled missing `embark_town` values with the mode.

### 2.2 Feature Engineering
- Added `family_size` = `sibsp` + `parch` + 1.
- Created binary feature `alone`.
- Encoded categorical variables (`sex` and `embarked`) as numeric.

  - **Rationale:**
    - Family size and traveling alone may influence survival probability.
    - Numeric encoding enables machine learning models to process these features.

---

## 3. Feature Selection and Target Definition
- Input features explored for modeling:  
  - **Case 1:** `alone`  
  - **Case 2:** `age`  
  - **Case 3:** `age` + `family_size`  

- Target variable: `survived`.

**Reasoning:**  
- Features are historically relevant to survival predictions.
- Multiple feature sets allow comparison of model performance and insight into which factors are most predictive.

---

## 4. Data Splitting and Stratification
- Split data into training and test sets using stratified sampling to preserve class balance.

**Analysis:**
- Stratified sampling ensures the training and test sets closely reflect the original class distribution.
- Improves model reliability by reducing bias and variance in predictions.

---

## 5. Model Training and Evaluation

### 5.1 Decision Tree Classifier
- Trained on all three feature sets.
- Evaluated using classification reports and confusion matrices.
- Visualized decision trees to interpret feature importance.

### 5.2 Support Vector Machine (SVM)
- Trained SVM models with different kernels (RBF, linear, polynomial, sigmoid).
- Visualized support vectors in 1D and 2D plots for insights into decision boundaries.

### 5.3 Neural Network Classifier
- Trained a Multi-Layer Perceptron using `age` and `family_size`.
- Evaluated performance on test data.
- Visualized decision surfaces to understand the model’s learned patterns.

**Reflection:**
- Comparing models helps identify which algorithm and feature combination best predicts survival.
- Decision Tree models are interpretable; SVMs handle complex boundaries; Neural Networks capture non-linear patterns.
- Visualizations provide insights into how models make predictions and where errors occur.

---

## 6. Final Thoughts & Insights
- Feature selection matters: even a simple feature like `alone` can provide predictive value.
- Multiple models reveal different strengths and weaknesses.
- Stratified sampling ensures reliable evaluation of model performance.
- Neural Networks perform well with two continuous features, capturing nuanced relationships.
- Future exploration could include hyperparameter tuning, additional features, or ensemble methods.

---

## How to Run

1. **Open the Project Notebook**  
   Navigate to `notebooks/project03` and open the Jupyter notebook:  
   `ml03_beaderstadt.ipynb`

2. **Select the Correct Kernel**  
   Ensure the notebook uses the correct Python environment where required libraries are installed.

3. **Clear Kernel / Outputs (Optional)**  
   Use Kernel -> Restart & Clear Outputs to start fresh and avoid stale variables or plots.

4. **Run the Notebook**  
   Execute cells sequentially to load data, prepare features, train models, and visualize results.

5. **View Results**  
   Classification reports, confusion matrices, decision trees, and decision surfaces display inline.  
   Use these outputs to analyze survival patterns and model performance.
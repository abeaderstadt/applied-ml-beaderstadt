# Project 2: Beaderstadt Titanic Data Exploration and Feature Engineering
> My personalized exploration and manipulation of the Titanic Dataset.

## Notebook

You can view the full Jupyter notebook for this project here:  
[ml01_beaderstadt.ipynb](https://github.com/abeaderstadt/applied-ml-beaderstadt/blob/main/notebooks/project02/ml02_beaderstadt.ipynb)

## Project Overview
In this project, I worked with the Titanic dataset to explore patterns, clean and transform the data, create new features, and prepare the dataset for potential machine learning models. The goal was to understand factors affecting survival rates and demonstrate a structured workflow for data analysis and feature preparation.

---

## Key Steps in This Project

## 1. Import and Inspect the Data
- Loaded the Titanic dataset directly from the seaborn library.

- Inspected the dataset using `.info()`, `.head()`, `.describe()`, and `.isnull().sum()` to check for missing values and data types.

- Explored correlations among numeric features.
  
  - **Insights:**
    - 891 data instances and 15 features.

    - Missing values in `age`, `deck`, and `embark_town`.

    - Several non-numeric features (categorical and boolean).

## 2. Data Exploration and Preparation

## 2.1 Visual Exploration
- Created scatter plots, scatter matrix, histograms, and count plots to examine distributions and relationships.

- Colored age vs fare scatter plots by gender to identify potential survival patterns.

  - **Observations:**

    - Certain classes and age groups had higher survival rates.

    - `pclass`, `sex`, and `age` stood out as potential predictors.

    - Visible imbalances in some categorical features like `pclass` and `survived`.

## 2.2 Handle Missing Values

- Imputed missing `age` values with the median.

- Filled missing `embark_town` values with the mode.

## 2.3 Feature Engineering

- Added `family_size` = `sibsp` + `parch` + 1.

- Converted categorical variables `sex` and `embarked` to numeric values.

- Created a binary feature `alone`.

  - **Rationale:**

    - Family size may influence survival probability.

    - Numeric encoding allows modeling and correlation analysis.
  

# 3. Feature Selection and Target Definition

- Selected input features: `age`, `fare`, `pclass`, `sex`, `family_size`.

- Target variable: `survived`.

**Reasoning:**

- These features are relevant to survival predictions based on historical analysis.

- `sex`, `pclass`, and `age` are likely the most predictive of survival.

# 4. Data Splitting and Stratification

- Split data into training and test sets using both basic train/test split and stratified sampling.

**Findings:**
| Set              | Non-survivors | Survivors |
| ---------------- | ------------- | --------- |
| Original         | 0.616         | 0.384     |
| Basic Train      | 0.611         | 0.389     |
| Basic Test       | 0.637         | 0.363     |
| Stratified Train | 0.617         | 0.383     |
| Stratified Test  | 0.615         | 0.385     |

**Analysis:**

- Why stratification improves performance: Stratified sampling preserves the original class distribution in both training and test sets, reducing potential bias and improving model reliability.

- Training/test distributions: Stratified split distributions closely match the original dataset, whereas the basic split shows slightly larger deviations.

- Better class balance: Stratified split provides a more consistent and representative balance for the target variable.

# 5. Conclusion

- This project demonstrates a structured workflow for exploring and preparing data for machine learning. Key takeaways:

  - Visual exploration identifies patterns and potential predictors.

  - Handling missing values and creating new features improves dataset quality.

  - Stratified splitting ensures training and test sets reflect the true distribution of the target.

---
# How to Run

## 1. Open the Project Notebook
Navigate to the notebooks/project02 folder and open the Jupyter notebook:

ml02_beaderstadt.ipynb

## 2. Select the Correct Kernel
Before running any cells, make sure the notebook is using the correct Python environment (kernel) where all required libraries are installed.
- In VS Code, check the top right of the notebook for the selected kernel.
- Switch to your project’s virtual environment if needed.

## 3. Clear Kernel / Outputs (Optional but Helpful)
If the notebook has previously been run, you can clear all outputs to start fresh:
- In VS Code or Jupyter, use Kernel -> Restart & Clear Outputs.
- This ensures that no stale variables or plots interfere with your current run.

## 4. Run the Notebook
In VS Code, ensure your Python environment is active.
Run each cell sequentially to execute the analysis, visualizations, and feature engineering.

## 5.View Results
Plots and tables will display inline in the notebook.
Check the final outputs for insights on survival patterns and feature preparation.


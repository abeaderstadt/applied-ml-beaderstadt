# Project 1: Beaderstadt California Housing Price Prediction
> My personalized exploration and manipulation of the California Housing Dataset.

## Notebook

You can view the full Jupyter notebook for this project here:  
[ml01_beaderstadt.ipynb](https://github.com/abeaderstadt/applied-ml-beaderstadt/blob/main/notebooks/project01/ml01.ipynb)


## Project Overview
In this project, I worked with the California housing dataset to uncover which factors have the biggest impact on home values. I’ll be building a predictive model and interpreting the results to see what they tell us about housing trends in California.

---

## Key Steps in This Project

## 1. Import and Inspect the Data
- Loaded the California housing dataset into a pandas DataFrame.

- Inspected the dataset using `.info()`, `.head()`, `.describe()`, and `.isnull().sum()` to check for missing values and data types.

- Explored correlations among numeric features.
  
  - **Insights:**
    - 20,640 data instances and 9 features..

    - All features are numeric; no missing values.

    - Strong predictors include `MedInc` (median income) and `AveRooms` (average rooms).

## 2. Data Exploration and Preparation

## 2.1 Visual Exploration
- Created histograms, boxplots, scatter plots, and a pairplot.

- Examined relationships between features and target variable (MedHouseVal).

  - **Observations:**

    - `MedInc` and `MedHouseVal` show a clear positive correlation.

    - `AveRooms` also contributes to house value prediction.

    - Some features (like `HouseAge`) are less predictive on their own.
  
    - Outliers exist in `AveRooms`, `AveBedrms`, and `Population`.

## 2.2 Feature Selection

- Selected input features: `MedInc` and `AveRooms`.

- Target variable: `MedHouseVal`.
  
  - **Reasoning:**

    - `MedInc` is the strongest driver of house prices.

    - `AveRooms` contributes additional context to household size and property value.


# 3. Data Splitting

- Split data into training and test sets (80% train / 20% test) using train_test_split.
```
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)
```
- Ensures reproducibility and fair evaluation of model performance.

# 4. Model Training and Evaluation
## 4.1 Train Linear Regression Model
- Trained a LinearRegression model using the training data.
```
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```
## 4.2 Model Performance Metrics

- R²: 0.46 - The model explains about 46% of the variation in house prices.

- MAE: 0.62 - Average prediction error of $62,000.

- RMSE: 0.84 - Typical prediction error of $84,000.

## 4.3 Actual vs Predicted Values

- Scatter plot shows predicted values mostly track actual values, with some spread for higher-priced homes.

# 5. Key Takeaways
- Median income (`MedInc`) is the most influential predictor.

- Average number of rooms (`AveRooms`) adds context but is less critical.

- Outliers in features can affect prediction accuracy.

- Adding more features (`house age`, `population`, `location`) could improve model performance.

- Linear regression provides a reasonable baseline but may be enhanced with more complex models.

---
# How to Run

## 1. Open the Project Notebook
Navigate to the notebooks/project01 folder and open the Jupyter notebook:

ml02.ipynb

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


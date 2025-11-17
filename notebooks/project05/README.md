# Project 5: Ensemble Machine Learning – Wine Dataset
> Exploring ensemble machine learning techniques.


## Project Overview
This project explores ensemble machine learning techniques to predict red wine quality using physicochemical properties. Ensemble methods combine multiple models to improve predictive performance, reduce overfitting, and generalize better to new data.

We focus on two ensemble approaches:

    1. Random Forest (100 trees) – a strong baseline with many decision trees.

    2. Voting Classifier (Decision Tree + SVM + Neural Network) – combines different model types to leverage their strengths.

The goal is to evaluate which approach predicts wine quality most accurately and understand why some models perform better than others.

---

### Dataset

- Source: UCI Wine Quality Dataset

- Description: 1599 red wine samples with 11 physicochemical features:

  - fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol

- Target: `quality_label` - categorical wine quality with three classes: low (3–4), medium (5–6), high (7–8)

We created two derived columns for modeling:

- `quality_label` (string labels)

- `quality_numeric` (0 = low, 1 = medium, 2 = high)

---

### Feature Selection

- **Target:** `quality_label`

- **Features:** All columns except `quality`, `quality_label`, and `quality_numeric`

- Reasoning: These 11 physicochemical properties are meaningful predictors of wine quality. Excluding any version of the target prevents data leakage.

---

### Data Preparation

- Converted the original numeric quality scores into categorical labels and numeric categories.

- Split the data into training (80%) and testing (20%) sets, stratifying by class to preserve class balance.

---

### Models Evaluated

- Random Forest (100 trees) – trains many trees in parallel and averages predictions.

- Voting Classifier (DT + SVM + NN) – combines three different model types for a soft-voting ensemble.

Evaluation metrics:

- Accuracy

- F1 Score (weighted)

- Confusion matrix for class-level performance

- Gap calculations: difference between training and testing performance to check for overfitting.

---

### Results Summary
| Model                  | Train Accuracy | Test Accuracy | Train F1 | Test F1 | Notes                                                   |
| ---------------------- | -------------- | ------------- | -------- | ------- | ------------------------------------------------------- |
| Random Forest (100)    | 1.0000         | 0.8875        | 1.0000   | 0.8661  | Slight overfit, best generalization                     |
| Voting (DT + SVM + NN) | 0.9234         | 0.8562        | 0.9054   | 0.8351  | More balanced training, slightly lower test performance |


**Key insights:**
- Random Forest performed better overall, achieving higher test accuracy and F1.
- Voting ensemble misclassified more of the hardest-to-predict wines (high-quality class).
- Gap analysis shows Random Forest slightly overfit, but still generalized well.

---

### Conclusions

- Random Forest (100) is the stronger model for predicting red wine quality in this dataset.

- Ensemble methods that combine multiple heterogeneous models (like Voting Classifier) are promising, but performance depends heavily on the individual models and their tuning.

- Next steps could include:

  - Hyperparameter tuning for both models

  - Feature importance analysis to identify key drivers of wine quality

  - Trying additional ensemble methods like Gradient Boosting or AdaBoost

---

## How to Run

1. **Open the Project Notebook**  
   Navigate to `notebooks/project05` and open the Jupyter notebook:  
   `ensemble-beaderstadt.ipynb`

2. **Select the Correct Kernel**  
   Ensure the notebook uses the correct Python environment where required libraries are installed.

3. **Clear Kernel / Outputs (Optional)**  
   Use Kernel -> Restart & Clear Outputs to start fresh and avoid stale variables or plots.

4. **Run the Notebook**  
   Execute cells sequentially to load data, prepare features, train models, and visualize results.

4. Run the notebook sections sequentially to load data, prepare it, train models, and view results.
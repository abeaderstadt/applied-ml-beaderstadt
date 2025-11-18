# Project 5: Ensemble Machine Learning – Wine Dataset
> Exploring ensemble machine learning techniques.


## Project Overview
This project uses the UCI Wine Quality dataset to explore two ensemble models: Random Forest (100) and a Voting model (DT + SVM + NN). The goal was to see which one performed better, understand why, and figure out whether combining different models actually leads to better predictions.

We focus on two ensemble approaches:

    1. Random Forest (100 trees) - a strong baseline with many decision trees.

    2. Voting Classifier (Decision Tree + SVM + Neural Network) - combines different model types to leverage their strengths.

The goal is to evaluate which approach predicts wine quality most accurately and understand why some models perform better than others.

---

### Dataset

- Source: UCI Wine Quality Dataset

- Description: 1599 wine samples with 11 features:

  - fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol

- Target: `quality_label` - (the new feature created) that has three classes: low (3-4), medium (5-6), high (7-8)

We created two additional columns for modeling:

- `quality_label` (string labels)

- `quality_numeric` (0 = low, 1 = medium, 2 = high)

---

### Feature Selection

- **Target:** `quality_label`

- **Features:**
   - `fixed acidity`         
   - `volatile acidity`      
   - `citric acid`          
   - `residual sugar`        
   - `chlorides`            
   - `free sulfur dioxide`  
   - `total sulfur dioxide`  
   - `density`               
   - `pH`                   
   - `sulphates`            
   - `alcohol`  

- **Reasoning:**
- Keeping all 11 features lets the ensemble models figure out which properties are most important on their own.
- Why the following columns were excluded:
  -  `quality`- original numeric score (we don’t want to leak the target)
  -  `quality_label`- that’s our target now
  -  `quality_numberic` - numeric version of the target 

---

### Data Preparation

- Converted the original numeric quality scores into categorical labels and numeric categories.

- Split the data into training (80%) and testing (20%) sets, stratifying by class to preserve class balance.

---

### Models Evaluated

- Random Forest (100 trees) - trains many trees in parallel and averages predictions.

- Voting Classifier (DT + SVM + NN) - combines three different model types for a soft-voting ensemble.

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
- Random Forest performed better overall, with higher test accuracy and F1.
- For both models most mistakes happen between adjacent classes, like confusing medium-quality wines for high or low.
- Gap analysis shows Random Forest slightly overfit, but still performed well.

---

### Conclusions

- Random Forest (100) is the stronger model for predicting wine quality.

- Using multiple different models together (like the Voting Classifier) can work really well, but only if each model is tuned properly. If one of them isn’t pulling its weight, the whole ensemble won’t perform as well

- Next steps could include:

  - Explore which features ( like alcohol and pH) consistently increase probability of high-quality label

  - Trying additional ensemble methods like Random Forest (200, max_depth=10)	which would add more trees, but limit tree depth to reduce overfitting
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
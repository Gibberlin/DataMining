# Assignment 3 — Iris Classification (KNN)

## Objective
Build a Machine Learning model to classify Iris flower species using the K-Nearest Neighbors (KNN) algorithm and use the trained model for prediction.

## Dataset
**Dataset Used:** Iris Dataset from `sklearn.datasets`

### Dataset Credit
The dataset used in this assignment was provided by **Syed Akhter Hussain**.

### Features
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

### Target Classes
| Class ID | Species |
|---------|---------|
| 0 | Setosa |
| 1 | Versicolor |
| 2 | Virginica |

## Implementation Overview

### Part 1 — Model Training (`iris_model_training.py`)

#### Step 1 — Load Dataset
Load the Iris dataset using `load_iris()` and extract:
- `X` → Features
- `y` → Labels

#### Step 2 — Split Dataset
Split the dataset into the following ratios:

| Set | Ratio |
|-----|-------|
| Training | 70% |
| Validation | 15% |
| Test | 15% |

#### Step 3 — Feature Scaling
Use `StandardScaler` to normalize the data for better KNN performance.

#### Step 4 — Train Model
Train the model using the K-Nearest Neighbors (KNN) algorithm with the following parameters:

| Parameter | Value |
|----------|-------|
| K | 5 |
| Distance Metric | Euclidean |
| Weights | Uniform |

#### Step 5 — Predictions
Make predictions on:
- Training set
- Validation set
- Test set

#### Step 6 — Performance Evaluation
Evaluate the model using the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

**Accuracy Formula:**

$$
\text{Accuracy} = \frac{\text{True Positive} + \text{True Negative}}{\text{Total Predictions}}
$$

#### Step 7 — Save Model
Save the trained artifacts as:
- `iris_knn_model.pkl`
- `iris_scaler.pkl`

#### Step 8 — Visualization
Create the following visualizations:
- Confusion Matrix (Heatmap)
- Accuracy Comparison (Train vs Validation vs Test)

Save the output figure as:
- `iris_model_performance.png`

#### Step 9 — Example Prediction
Test the model with sample inputs:
- `[5.0, 3.5, 1.3, 0.3]` → **Setosa**
- `[6.5, 3.0, 5.5, 1.8]` → **Virginica**

### Part 2 — Model Usage (`use_iris_model.py`)

#### Step 1 — Load Model
Load the following saved files:
- `iris_knn_model.pkl`
- `iris_scaler.pkl`

#### Step 2 — Prediction Methods

##### Method 1 — Individual Prediction
Predict a single flower using:
- **Input:** `[Sepal Length, Sepal Width, Petal Length, Petal Width]`
- **Output:** Predicted species and confidence

##### Method 2 — Batch Prediction (CSV)
Load data from:
- `iris_data/iris.csv`

Predict all rows and save the results as:
- `iris_predictions.csv`

##### Method 3 — Interactive Prediction
Allow manual user input such as:
- `5.1,3.5,1.4,0.2`

Output:
- Predicted species
- Probability distribution


##  Sample Output


![Assignment 3 Output](../Images/assing3.png)
---

## Features
- Uses real dataset
- End-to-end machine learning pipeline
- Model saving and loading
- Visualization included
- Multiple prediction methods
- Beginner-friendly

## Limitations
- Sensitive to feature scaling
- KNN is slower for large datasets
- Performance depends on the value of `K`

## Future Improvements
- Hyperparameter tuning for K optimization
- Use cross-validation
- Try other models such as SVM and Random Forest
- Build a web interface using Flask or React

## Author
**Syed Yashin Hussain**  
Roll Number: 24205007014  
B.Tech CSE Student  
Barak Valley Engineering College

# Assignment on Data-Mining

## Submission Details

| Field | Details |
|-------|---------|
| **Subject Name** | Data-Mining |
| **College Name** | Barak Valley Engineering College |
| **Submitted By** | Syed Yashin Hussain |
| **Roll Number** | 24205007014 |
| **Submitted To** | Mr. SHASHADAR DAS |

***

## Overview
This repository contains multiple **Data Mining assignments** covering machine learning, deep learning, and probability-based classification techniques. Each assignment demonstrates a different approach to solving real-world classification problems using practical datasets and structured workflows.

***

# Assignment 1 — Digit Detection without Neural Networks

## Objective
Build a model to detect digits **without using Neural Networks** by applying the **K-Nearest Neighbors (KNN)** algorithm.

## Dataset Credit
The dataset used in this assignment was provided by **Syed Akhter Hussain**.

## Implementation Steps

### Step 1 — Download Dataset
Download the dataset from Kaggle.

### Step 2 — Split Dataset
Split the dataset into:
- **Training:** 80%
- **Validation:** 10%
- **Test:** 10%

using the `split-folders` package.

### Step 3 — Train Model
Train the model in Google Colab using the **KNN algorithm** with **Euclidean distance**.

| Parameter | Value |
|-----------|-------|
| K value | 5 |
| Distance Metric | Euclidean |

### Training Logic
- Define `x` and `y` arrays for image data and labels
- Load images into `x`
- Load folder labels into `y`
- Convert image data into NumPy arrays
- Train the model using `KNeighborsClassifier()`
- Fit the training data and labels
- Save the trained model in `.pkl` format

### Step 4 — Prediction
Load the saved model and use it for prediction.

### Step 5 — Accuracy Calculation
Accuracy formula:

$$
\text{Accuracy} = \frac{\text{True Positive} + \text{True Negative}}{\text{Total Predictions}}
$$

### Sample Output

![Assignment 1 Output](Images/assing1.png)

***

# Assignment 2 — Spam Detection (Naive Bayes)

## Objective
Build a **Spam Classifier** using the **Naive Bayes algorithm** without using machine learning libraries.

## Dataset
**File:** `dataset.csv`

### Dataset Credit
The dataset used in this assignment was provided by **Syed Akhter Hussain**.

### Sample Format
| word | not_spam | spam |
|------|----------|------|
| hello | 10 | 2 |
| offer | 1 | 15 |

## Implementation Logic

### Load Dataset
- Read CSV data using Python's `csv` module
- Store word frequencies for:
  - Spam
  - Not Spam

### Probability Calculation
Compute prior probabilities:
- `P(Not Spam)`
- `P(Spam)`

### Classification Logic
Apply **Laplace Smoothing**:

$$
P(\text{word} \mid \text{class}) = \frac{\text{count} + 1}{\text{total words} + \text{vocab size}}
$$

### Naive Bayes Formula
$$
P(C \mid X) = \frac{P(X \mid C) \cdot P(C)}{P(X)}
$$

Where:
- `C` = Class (Spam / Not Spam)
- `X` = Input message

### Prediction
- Calculate class probabilities
- Compare the results
- Output either:
  - `SPAM`
  - `NOT SPAM`

### Sample Output

![Assignment 2 Output](Images/assing2.png)

***

# Assignment 3 — Iris Classification (KNN)

## Objective
Build a machine learning model to classify Iris flower species using the **K-Nearest Neighbors (KNN)** algorithm and use the trained model for prediction.

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
Split the dataset into:

| Set | Ratio |
|-----|-------|
| Training | 70% |
| Validation | 15% |
| Test | 15% |

#### Step 3 — Feature Scaling
Use `StandardScaler` to normalize the data.

#### Step 4 — Train Model
Use KNN with:

| Parameter | Value |
|----------|-------|
| K | 5 |
| Distance Metric | Euclidean |
| Weights | Uniform |

#### Step 5 — Predictions
Predict on:
- Training set
- Validation set
- Test set

#### Step 6 — Evaluation
Metrics used:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

#### Step 7 — Save Model
Save:
- `iris_knn_model.pkl`
- `iris_scaler.pkl`

#### Step 8 — Visualization
Create:
- Confusion Matrix (Heatmap)
- Accuracy comparison chart

Save as:
- `iris_model_performance.png`

#### Step 9 — Example Prediction
- `[5.0, 3.5, 1.3, 0.3]` → **Setosa**
- `[6.5, 3.0, 5.5, 1.8]` → **Virginica**

### Part 2 — Model Usage (`use_iris_model.py`)

#### Prediction Methods
- Individual prediction
- Batch prediction using CSV
- Interactive manual prediction

### Sample Output

![Assignment 3 Output](Images/assing3.png)

***

# Assignment 4 — Skin Cancer Detection (CNN - MobileNetV2)

## Objective
Build a deep learning model to classify skin images into:
- **Melanoma (Cancer)**
- **Not Melanoma (Non-Cancer)**

using **Transfer Learning with MobileNetV2**.

## Dataset Structure
```text
DermMel/
├── train_sep/
│   ├── Melanoma/
│   └── NotMelanoma/
├── valid/
└── test/
```

### Dataset Credit
The dataset used in this assignment was provided by **Syed Akhter Hussain**.

### Dataset Notes
- Images loaded using `flow_from_directory()`
- Classes automatically detected from folder names
- Preprocessing applied using `preprocess_input`

## Implementation Logic

### Data Augmentation
Applied on training data:
- Rotation
- Zoom
- Horizontal Flip

### Class Weights
Use `compute_class_weight()` to handle imbalance.

### Model Architecture
- Base Model: **MobileNetV2** pretrained on ImageNet
- GlobalAveragePooling2D
- Dense (128, ReLU)
- BatchNormalization
- Dropout (0.4)
- Output layer (Softmax, 2 classes)

### Training Strategy
#### Stage 1 — Transfer Learning
- Freeze base model
- Train top layers only

#### Stage 2 — Fine Tuning
- Unfreeze last 30 layers
- Reduce learning rate
- Improve performance

### Callbacks
- `EarlyStopping`
- `ReduceLROnPlateau`
- `ModelCheckpoint`

### Model Saving
- `cancer_model.h5`
- `best_model.h5`

### Evaluation
- Confusion Matrix
- Classification Report
- Accuracy
- Precision
- Recall
- F1 Score

### Prediction Input Example
`DermMel/test/Melanoma/AUG_0_11.jpeg`

### Sample Prediction Output
```text
Prediction: Melanoma
Confidence: 97.45%
```

### Sample Output Images

![Assignment 4 Output 1](Images/assing4_output1.png)

![Assignment 4 Output 2](Images/assing4_output2.png)


### Confusion Matrix

![Assignment 4 Confusion Matrix](Images/assing4_confusion_matrix.png)

***

## Common Features
- Real-world datasets
- Machine learning and deep learning workflows
- Model training and prediction
- Visualization support
- Beginner-friendly project structure
- Practical classification tasks

## Author
**Syed Yashin Hussain**  
Roll Number: 24205007014  
B.Tech CSE Student  
Barak Valley Engineering College

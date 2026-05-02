<!-- A Data Mining assignment demonstrating data cleaning, exploratory analysis, and predictive modeling.
# Assignment 1
Question: Detect Digit without using Neural Networks

## Dataset Credit
The dataset used in this assignment was provided by **Syed Akhter Hussain**.

Implement:
Step 1: Download dataset from kaggle
Step 2: Split Dataset in Val,Test,Train with the Ratio of 10,10,80 by using split-folders package
Step 3: Then train model in google colab 
        Where used k neighbour technique 
        In this technique used euclidean method 
            a: Where set k value = 5
            b: Where set x,y array for load image and folder eg:1-9
            c: Then load then images folder in y and images in x 
            d: Loaded images save as a array using numpy array
            e: Then define knn using KNeighorsClassifier(which is an function provided by sklearn) with the parameter K which set previous(K means number of neighbors to be compare with) for classified images in classes.
            f: Fit the images in training folder and images.
                Fit means compare the images for training folder with there respective folder label eg: if one image is found in its respective folder name is 1 then its Class declare as 1.
            g: compare every images and there folder label and save data and export as .pkl format
Step 4: After import model implement it to predict
Step 5: Calculate Validation and Test Accuracy by 
        # TRUE POSITIVE + FALSE POSITIVE / TOTAL PREDICTION
        a:  If system predict a image is 1 and its actually 1 then its TRUE POSITIVE 
        b: If system predict a image is 1 and its actually not 1 then its FALSE POSITIVE
        c: If system predict a image is not 1 and its actually 1 then its FALSE NEGATIVE
        d: If system predict a image is not 1 and its actually not 1 then its TRUE NEGATIVE
        e: And all TRUE POSITIVE + FALSE POSITIVE + FALSE NEGATIVE + TRUE NEGATIVE occure during prediction call as TOTAL PREDICTION
 -->



# Data-Mining-Assignment-6th-Sem

A Data Mining assignment demonstrating data cleaning, exploratory analysis, and predictive modeling.

## Student Details

| Field | Details |
|-------|---------|
| Name | Syed Yashin Hussain |
| Roll Number | 24205007014 |
| Course | B.Tech CSE |
| College | Barak Valley Engineering College |


# Assignment 1

**Question:** Detect Digit without using Neural Networks

## Dataset Credit
The dataset used in this assignment was provided by **Syed Akhter Hussain**.

## Implementation Steps

### Step 1 — Download Dataset
Download the dataset from Kaggle.

---

### Step 2 — Split Dataset
Split the dataset into **Val / Test / Train** with the ratio of **10 / 10 / 80** using the `split-folders` package.

---

### Step 3 — Train Model (Google Colab)
Train the model using the **K-Nearest Neighbors (KNN)** technique with the **Euclidean** distance method.

| Parameter | Value |
|-----------|-------|
| K value | 5 |
| Distance metric | Euclidean |

**How it works:**

- **a.** Set `K = 5` — this means the model compares each image to its 5 nearest neighbors
- **b.** Define `x` and `y` arrays to hold images and their folder labels (e.g. folders 1–9)
- **c.** Load images into `x` and their folder names into `y`
- **d.** Save the loaded images as a **NumPy array**
- **e.** Define KNN using `KNeighborsClassifier()` from `sklearn` — the `K` parameter sets how many neighbors to compare against for classifying an image
- **f.** Fit the training images and labels
  > *Fit means: compare every image with its folder label. For example, if an image is found in folder `1`, its class is declared as `1`*
- **g.** Compare every image with its folder label, save the result, and export as **`.pkl`** format

---

### Step 4 — Predict
Import the saved `.pkl` model and use it to predict new images.

---

### Step 5 — Calculate Validation and Test Accuracy

**Formula:**
```
Accuracy = (TRUE POSITIVE + TRUE NEGATIVE) / TOTAL PREDICTIONS
```

| Term | Meaning |
|------|---------|
| **TRUE POSITIVE** | System predicts image is `1`, and it actually is `1`. |
| **FALSE POSITIVE** | System predicts image is `1`, but it actually is not `1`. |
| **FALSE NEGATIVE** | System predicts image is not `1`, but it actually is `1`. |
| **TRUE NEGATIVE** | System predicts image is not `1`, and it actually is not `1`. |

 **TOTAL PREDICTIONS** = TRUE POSITIVE + FALSE POSITIVE + FALSE NEGATIVE + TRUE NEGATIVE  
 (all outcomes that occur during prediction)

 ![Assignment 1 Output](../Images/assing1.png)


##  Author

**Syed Yashin Hussain**  
Roll Number: 24205007014  
B.Tech CSE Student  
Barak Valley Engineering College  

---

# Assignment 2 — Spam Detection (Naive Bayes)

## Objective
Build a **Spam Classifier** using the **Naive Bayes algorithm** *without using any machine learning libraries*.

---

## Dataset

**File:** `dataset.csv`

### Dataset Credit
The dataset used in this assignment was provided by **Syed Akhter Hussain**.

### Format:
| word  | not_spam | spam |
|------|----------|------|
| hello | 10       | 2    |
| offer | 1        | 15   |

---

## Implementation Logic

### Load Dataset
- Read CSV file using Python `csv` module  
- Store word frequencies for both classes:
  - Spam
  - Not Spam  

---

### Calculate Probabilities
Compute prior probabilities:

```
P(Not Spam)
P(Spam)
```

---

### Classification Logic

Apply **Laplace Smoothing**:

```
P(word | class) = (count + 1) / (total_words + vocab_size)
```

---

### Naive Bayes Formula

```
P(C | X) = (P(X | C) * P(C)) / P(X)
```

Where:
- `C` = Class (Spam / Not Spam)
- `X` = Input message

---

### Prediction
- Calculate probabilities for both classes  
- Compare results  
- Output classification:

```
SPAM
or
NOT SPAM
```


## Sample Output


![Assignment 2 Output](../Images/assing2.png)
---

## Features

- No external machine learning libraries
- Lightweight implementation
- Beginner-friendly structure
- Based on probability concepts

---

## Limitations

- Assumes **word independence**
- Accuracy depends on dataset quality
- Not suitable for very large datasets

---

## Future Improvements

- Add text preprocessing (stopwords, stemming)
- Use TF-IDF
- Build web interface
- Compare multiple algorithms

---

## Author

**Syed Yashin Hussain**  
Roll Number: 24205007014  
B.Tech CSE Student  
Barak Valley Engineering College  

---

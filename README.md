# Data Mining Assignments

<p>
  <img src="https://img.shields.io/badge/Subject-Data%20Mining-2F80ED" alt="Data Mining">
  <img src="https://img.shields.io/badge/Projects-4-27AE60" alt="4 Projects">
  <img src="https://img.shields.io/badge/Python-3.10%2B-F2994A" alt="Python">
  <img src="https://img.shields.io/badge/Status-Organized-9B51E0" alt="Organized">
</p>

## Submission Details

| Field | Details |
| --- | --- |
| **Subject Name** | Data Mining |
| **College Name** | Barak Valley Engineering College |
| **Submitted By** | Syed Yashin Hussain |
| **Roll Number** | 242050007014 |
| **Submitted To** | Mr. SHASHADAR DAS |

## Overview

This repository contains four Data Mining assignments covering **KNN**, **Naive Bayes**, and **CNN transfer learning**. Each assignment now has its own README with prerequisites, run commands, dataset paths, and usage notes.

## Project Index

| Project | Topic | Main Files | README |
| --- | --- | --- | --- |
| <span style="color:#2F80ED"><b>Assignment 1</b></span> | Digit detection without neural networks | `Assignment_1.ipynb` | [Open](Assignment-1/README.md) |
| <span style="color:#9B51E0"><b>Assignment 2</b></span> | Spam detection with Naive Bayes | `naive_bayes_spam.py` | [Open](Assignment-2/README.md) |
| <span style="color:#27AE60"><b>Assignment 3</b></span> | Iris classification with KNN | `iris_model_training.py`, `use_iris_model.py` | [Open](Assignment-3/README.md) |
| <span style="color:#EB5757"><b>Assignment 4</b></span> | Skin cancer image classification | `train_model.py`, `predict.py` | [Open](Assignment-4/README.md) |

## Repository Structure

```text
DataMining/
├── Assignment-1/
│   ├── Assignment_1.ipynb
│   ├── README.md
│   └── knn_model.pkl
├── Assignment-2/
│   ├── naive_bayes_spam.py
│   └── README.md
├── Assignment-3/
│   ├── iris_model_training.py
│   ├── use_iris_model.py
│   ├── README.md
│   └── iris_predictions.csv
├── Assignment-4/
│   ├── train_model.py
│   ├── predict.py
│   └── README.md
├── Datasets/
│   ├── Assignment-1/
│   ├── Assignment-2/
│   ├── Assignment-3/
│   └── Assignment-4/
└── Images/
```

## Dataset Layout

Datasets are kept in one top-level folder so the assignment folders stay clean:

| Assignment | Dataset Path |
| --- | --- |
| Assignment 1 | `Datasets/Assignment-1/dataset/` and `Datasets/Assignment-1/dataset_split/` |
| Assignment 2 | `Datasets/Assignment-2/dataset.csv` |
| Assignment 3 | `Datasets/Assignment-3/iris_data/iris.csv` |
| Assignment 4 | `Datasets/Assignment-4/DermMel/` |

## Quick Start

Install the packages needed by the project you want to run:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn pillow tensorflow notebook opencv-python split-folders kagglehub
```

Run an assignment:

```bash
jupyter notebook Assignment-1/Assignment_1.ipynb
python Assignment-2/naive_bayes_spam.py
python Assignment-3/iris_model_training.py
python Assignment-3/use_iris_model.py
python Assignment-4/train_model.py
python Assignment-4/predict.py
```

## Dataset Credit

The datasets used in these assignments were provided by **Syed Akhter Hussain**. Assignment 1 also references the Kaggle digit dataset used in the notebook.

## Common Features

<span style="color:#27AE60"><b>Implemented:</b></span> real datasets, model training, prediction workflows, saved models, metrics, and visual outputs.

<span style="color:#F2994A"><b>Recommended:</b></span> run each assignment from the repository root so relative paths match the README examples.

## Author

**Syed Yashin Hussain**  
Roll Number: **242050007014**  
B.Tech CSE Student  
Barak Valley Engineering College

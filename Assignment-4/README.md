# Assignment 4 — Skin Cancer Detection (CNN - MobileNetV2)

## Objective
Build a deep learning model to classify skin lesion images into the following two categories:

- **Melanoma (Cancer)**
- **Not Melanoma (Non-Cancer)**

The project uses **Transfer Learning with MobileNetV2** to improve classification accuracy and training efficiency.

## Dataset
The dataset is organized into separate folders for training, validation, and testing.

### Dataset Credit
The dataset used in this assignment was provided by **Syed Akhter Hussain**.

```text
DermMel/
├── train_sep/
│   ├── Melanoma/
│   └── NotMelanoma/
├── valid/
└── test/
```

### Dataset Notes
- Images are loaded using `flow_from_directory()`.
- Class labels are automatically assigned from folder names.
- Preprocessing is applied using `preprocess_input` from MobileNetV2.

## Implementation Logic

### Load Dataset
The dataset is loaded using `ImageDataGenerator`, which handles image loading, preprocessing, and augmentation.

### Data Augmentation
The following augmentation techniques are applied to training images:
- Rotation
- Zoom
- Horizontal flip

These transformations help improve model generalization and reduce overfitting.

### Class Weights
To handle class imbalance, class weights are calculated using:

```python
compute_class_weight()
```

This ensures that minority classes receive appropriate importance during training.

## Model Architecture
The model is built using **MobileNetV2** pretrained on ImageNet as the base model.

### Base Model
- **MobileNetV2** (pretrained on ImageNet)

### Custom Layers
- GlobalAveragePooling2D
- Dense layer with 128 units and ReLU activation
- BatchNormalization
- Dropout with rate 0.4
- Output layer with Softmax activation for 2 classes

## Training Strategy

### Stage 1 — Transfer Learning
- Freeze all layers of the base MobileNetV2 model
- Train only the custom classification head

### Stage 2 — Fine Tuning
- Unfreeze the last 30 layers of the base model
- Reduce the learning rate
- Continue training to improve feature adaptation and performance

## Callbacks
The following callbacks are used during training:

- `EarlyStopping` — Stops training when validation performance stops improving
- `ReduceLROnPlateau` — Reduces learning rate when progress stalls
- `ModelCheckpoint` — Saves the best model during training

## Model Saving
The trained models are saved as:

- `cancer_model.h5` — Final trained model
- `best_model.h5` — Best-performing model based on validation results

## Evaluation
The model is evaluated using the following metrics:

- Confusion Matrix
- Classification Report
- Accuracy
- Precision
- Recall
- F1 Score

### Accuracy Formula
$$
\text{Accuracy} = \frac{\text{True Positive} + \text{True Negative}}{\text{Total Predictions}}
$$

## Prediction Script (`Predict.py`)

### Input
Example input image path:

```text
DermMel/test/Melanoma/AUG_0_11.jpeg
```

### Prediction Logic
The prediction script performs the following steps:

1. Loads the trained model from `cancer_model.h5`
2. Resizes the image to `224 x 224`
3. Applies MobileNetV2 preprocessing
4. Predicts the class label
5. Displays the confidence score

### Sample Output
```text
Prediction: Melanoma
Confidence: 97.45%
```

## Sample Output Images

![Assignment 4 Output 1](../Images/assing4_output1.png)

![Assignment 4 Output 2](../Images/assing4_output2.png)


## Confusion Matrix
![Assignment 4 Confusion Matrix](../Images/assing4_confusion_matrix.png)

## Features
- Deep learning model using CNN
- Transfer learning with MobileNetV2
- Data augmentation support
- Class imbalance handling
- Model saving and loading
- Image-based prediction system

## Limitations
- Requires good dataset quality
- Needs GPU for faster training
- May overfit on small datasets
- Performance depends on class balance and dataset diversity

## Future Improvements
- Add Grad-CAM visualization for explainability
- Use larger and more diverse datasets
- Build a web interface for prediction
- Try advanced architectures such as EfficientNet

## Author
**Syed Yashin Hussain**  
Roll Number: 24205007014  
B.Tech CSE Student  
Barak Valley Engineering College


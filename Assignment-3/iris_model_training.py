import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'iris_knn_model.pkl'
SCALER_PATH = BASE_DIR / 'iris_scaler.pkl'
PERFORMANCE_IMAGE_PATH = BASE_DIR / 'iris_model_performance.png'

# Load Iris dataset
print("Loading Iris dataset...")
iris = load_iris()
X = iris.data  # Features (Sepal Length, Sepal Width, Petal Length, Petal Width)
y = iris.target  # Target (Species: 0=Setosa, 1=Versicolor, 2=Virginica)

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Features: {iris.feature_names}")
print(f"Classes: {iris.target_names}\n")

# Split dataset into Train (70%), Validation (15%), Test (15%)
print("Splitting dataset into Train/Validation/Test with ratio 70:15:15...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation set size: {X_val.shape[0]} ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)\n")

# Standardize the features (important for KNN)
print("Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train KNN model with K=5 (similar to Assignment 1)
print("Training KNN model with K=5...")
knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='uniform')
knn_model.fit(X_train_scaled, y_train)

# Make predictions on Validation and Test sets
print("Making predictions...\n")
y_val_pred = knn_model.predict(X_val_scaled)
y_test_pred = knn_model.predict(X_test_scaled)

# Calculate accuracies
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
train_accuracy = accuracy_score(y_train, knn_model.predict(X_train_scaled))

print("=" * 50)
print("MODEL PERFORMANCE METRICS")
print("=" * 50)
print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n")

# Detailed metrics for Test set
print("Detailed Test Set Metrics:")
print(f"Precision: {precision_score(y_test, y_test_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred, average='weighted'):.4f}")
print(f"F1-Score: {f1_score(y_test, y_test_pred, average='weighted'):.4f}\n")

# Classification Report
print("Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=iris.target_names))

# Confusion Matrix
print("\nConfusion Matrix (Test Set):")
cm_test = confusion_matrix(y_test, y_test_pred)
print(cm_test)

# Save model and scaler
print("\nSaving model and scaler...")
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(knn_model, f)
print(f"Model saved as '{MODEL_PATH.name}'")

with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved as '{SCALER_PATH.name}'")

# Visualization
print("\nGenerating visualizations...\n")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix Heatmap
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title('Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Accuracy Comparison
accuracies = [train_accuracy, val_accuracy, test_accuracy]
sets = ['Training', 'Validation', 'Test']
colors = ['#2ecc71', '#3498db', '#e74c3c']
axes[1].bar(sets, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[1].set_ylabel('Accuracy', fontsize=11)
axes[1].set_title('Model Accuracy Across Sets', fontsize=12, fontweight='bold')
axes[1].set_ylim([0, 1.0])
axes[1].grid(axis='y', alpha=0.3)

# Add percentage labels on bars
for i, (acc, set_name) in enumerate(zip(accuracies, sets)):
    axes[1].text(i, acc + 0.02, f'{acc*100:.2f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(PERFORMANCE_IMAGE_PATH, dpi=300, bbox_inches='tight')
print(f"Performance visualization saved as '{PERFORMANCE_IMAGE_PATH.name}'")
plt.show()

# Function to load and use the model for predictions
def load_and_predict(features):
    """
    Load the trained model and make predictions on new data.

    Args:
        features: numpy array of shape (n_samples, 4) with iris features
                 [Sepal Length, Sepal Width, Petal Length, Petal Width]

    Returns:
        predictions: numpy array of predicted class indices
        species_names: list of predicted species names
    """
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    features_scaled = scaler.transform(features)
    predictions = model.predict(features_scaled)
    species_names = [iris.target_names[pred] for pred in predictions]

    return predictions, species_names

# Example: Make prediction on new data
print("\n" + "=" * 50)
print("EXAMPLE: Making predictions on new data")
print("=" * 50)
example_features = np.array([[5.0, 3.5, 1.3, 0.3],    # Likely Setosa
                              [6.5, 3.0, 5.5, 1.8],    # Likely Virginica
                              [6.0, 2.7, 5.1, 1.6]])   # Likely Versicolor

predictions, species = load_and_predict(example_features)
for feat, pred_class, pred_species in zip(example_features, predictions, species):
    print(f"\nFeatures: {feat}")
    print(f"Predicted Species: {pred_species}")

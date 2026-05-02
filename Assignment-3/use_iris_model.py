import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_iris

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'iris_knn_model.pkl'
SCALER_PATH = BASE_DIR / 'iris_scaler.pkl'
DATA_PATH = BASE_DIR / 'iris_data' / 'iris.csv'
PREDICTIONS_PATH = BASE_DIR / 'iris_predictions.csv'

# Load the trained model and scaler
print("Loading trained model and scaler...")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# Get Iris dataset for reference
iris = load_iris()

print("[OK] Model loaded successfully!")
print(f"Model type: {type(model).__name__}")
print(f"Number of neighbors (K): {model.n_neighbors}")
print()

# ============================================================
# Method 1: Predict individual flowers
# ============================================================
print("="*60)
print("METHOD 1: Predicting Individual Flowers")
print("="*60)

# Example iris flowers with features:
# [Sepal Length, Sepal Width, Petal Length, Petal Width]

sample_flowers = [
    {
        'name': 'Sample 1',
        'features': [5.0, 3.5, 1.3, 0.3],
        'actual_type': 'Setosa'
    },
    {
        'name': 'Sample 2',
        'features': [6.5, 3.0, 5.5, 1.8],
        'actual_type': 'Virginica'
    },
    {
        'name': 'Sample 3',
        'features': [6.0, 2.7, 5.1, 1.6],
        'actual_type': 'Versicolor'
    },
    {
        'name': 'Sample 4',
        'features': [4.8, 3.4, 1.6, 0.2],
        'actual_type': 'Setosa'
    }
]

for sample in sample_flowers:
    features = np.array(sample['features']).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    predicted_species = iris.target_names[prediction]

    print(f"\n{sample['name']}:")
    print(f"  Features: Sepal Length={sample['features'][0]}, Sepal Width={sample['features'][1]}, "
          f"Petal Length={sample['features'][2]}, Petal Width={sample['features'][3]}")
    print(f"  Predicted Species: {predicted_species}")
    print(f"  Confidence: {probabilities[prediction]*100:.2f}%")
    print(f"  Actual Type: {sample['actual_type']}")
    match_result = "Correct" if predicted_species.capitalize() == sample['actual_type'].lower().replace('-', ' ').title() else "Incorrect"
    print(f"  Match: {match_result}")

# ============================================================
# Method 2: Batch predictions from DataFrame
# ============================================================
print("\n" + "="*60)
print("METHOD 2: Batch Predictions from CSV")
print("="*60)

# Load test data
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Batch prediction CSV file not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"\nLoaded {len(df)} samples from {DATA_PATH}")

# Extract features and drop Id column
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y_actual = df['Species'].values

# Make predictions
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)

# Convert predictions to species names
predicted_species = [iris.target_names[pred] for pred in predictions]

# Create results dataframe
results_df = pd.DataFrame({
    'Sepal_Length': X[:, 0],
    'Sepal_Width': X[:, 1],
    'Petal_Length': X[:, 2],
    'Petal_Width': X[:, 3],
    'Actual_Species': y_actual,
    'Predicted_Species': predicted_species,
})

# Calculate accuracy
accuracy = np.mean(predictions == np.array([0 if 'setosa' in s.lower() else 1 if 'versicolor' in s.lower() else 2 for s in y_actual]))
print(f"\nPrediction Accuracy: {accuracy*100:.2f}%")
print("\nFirst 10 predictions:")
print(results_df.head(10))

# Save results
results_df.to_csv(PREDICTIONS_PATH, index=False)
print(f"\n[OK] Full predictions saved to: {PREDICTIONS_PATH}")

# ============================================================
# Method 3: Interactive prediction
# ============================================================
print("\n" + "="*60)
print("METHOD 3: Interactive Single Prediction")
print("="*60)

print("\nEnter iris flower measurements (or 'q' to quit):")
print("Features needed: Sepal Length, Sepal Width, Petal Length, Petal Width\n")

while True:
    user_input = input("Enter measurements (comma-separated) or 'q' to quit: ").strip()

    if user_input.lower() == 'q':
        print("Exiting...")
        break

    try:
        values = [float(x.strip()) for x in user_input.split(',')]
        if len(values) != 4:
            print(f"Error: Expected 4 values, got {len(values)}")
            continue

        features = np.array(values).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        predicted_species = iris.target_names[prediction]

        print(f"\nPredicted Species: {predicted_species.upper()}")
        print("Confidence distribution:")
        for i, prob in enumerate(probabilities):
            print(f"  - {iris.target_names[i].capitalize()}: {prob*100:.2f}%")
        print()

    except ValueError:
        print("Error: Please enter 4 numeric values separated by commas (e.g., 5.0,3.5,1.3,0.3)")
        continue

print("\n" + "="*60)
print("Model usage examples completed!")
print("="*60)

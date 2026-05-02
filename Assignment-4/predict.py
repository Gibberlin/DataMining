import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
MODEL_PATH = BASE_DIR / "cancer_model.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names.txt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

if CLASS_NAMES_PATH.exists():
    class_names = CLASS_NAMES_PATH.read_text(encoding="utf-8").splitlines()
else:
    class_names = ['Melanoma', 'NotMelanoma']

def predict_image(image_path):
    image_path = Path(image_path)
    if not image_path.is_absolute():
        image_path = ROOT_DIR / image_path
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]

    class_index = np.argmax(preds)
    label = class_names[class_index]
    confidence = preds[class_index]

    print(f"\nImage: {image_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence*100:.2f}%")

    return label, confidence


if __name__ == "__main__":
    image_path = "Datasets/Assignment-4/DermMel/test/NotMelanoma/ISIC_0024307.jpg"
    predict_image(image_path)

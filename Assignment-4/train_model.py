import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# ========================
# CONFIG
# ========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 20

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATASET_DIR = ROOT_DIR / "Datasets" / "Assignment-4" / "DermMel"
train_dir = DATASET_DIR / "train_sep"
val_dir = DATASET_DIR / "valid"
test_dir = DATASET_DIR / "test"
MODEL_PATH = BASE_DIR / "cancer_model.h5"
BEST_MODEL_PATH = BASE_DIR / "best_model.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names.txt"
CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix.png"

for required_dir in [train_dir, val_dir, test_dir]:
    if not required_dir.exists():
        raise FileNotFoundError(f"Required dataset directory not found: {required_dir}")

# ========================
# DATA GENERATORS
# ========================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ========================
# CLASS INFO
# ========================
class_names = list(train_data.class_indices.keys())
print("Class names:", class_names)
CLASS_NAMES_PATH.write_text("\n".join(class_names), encoding="utf-8")

# ========================
# CLASS WEIGHTS
# ========================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# ========================
# MODEL
# ========================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer="adam",
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ========================
# CALLBACKS
# ========================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    str(BEST_MODEL_PATH),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# ========================
# TRAINING STAGE 1
# ========================
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_STAGE1,
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weights
)

# ========================
# FINE-TUNING STAGE
# ========================

# Freeze all layers first
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze last 30 layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_STAGE2,
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weights
)

# ========================
# SAVE MODEL
# ========================
model.save(str(MODEL_PATH))

# ========================
# EVALUATION
# ========================
raw_preds = model.predict(test_data)
print("Sample raw predictions:", raw_preds[:5])

y_pred = np.argmax(raw_preds, axis=1)
y_true = test_data.classes

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ========================
# HEATMAP
# ========================
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.savefig(CONFUSION_MATRIX_PATH)
plt.show()

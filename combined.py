# =======================
# IMPORT LIBRARY
# =======================
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# =======================
# KONFIGURASI DATA & MODEL
# =======================
dataset_dir = "dataset_combined"     # Direktori dataset
image_size = (224, 224)              # Ukuran gambar
batch_size = 32                      # Ukuran batch
num_classes = 6                      # Jumlah kelas
epochs = 20                          # Jumlah epoch

# =======================
# DATA AUGMENTATION
# =======================
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# Generator untuk training
train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

# Generator untuk validasi
val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# =======================
# INFO LABEL
# =======================
print("Mapping Kelas:")
for label, idx in train_gen.class_indices.items():
    print(f"{idx}: {label}")

# =======================
# BANGUN MODEL (MobileNetV2)
# =======================
base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights="imagenet")

# Tambahkan top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

# =======================
# KOMPILASI MODEL
# =======================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =======================
# CALLBACKS
# =======================
checkpoint = ModelCheckpoint(
    "model_combined.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# =======================
# TRAINING MODEL
# =======================
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=[checkpoint, earlystop]
)

print("✅ Model selesai dilatih dan disimpan sebagai 'model_combined.h5'")

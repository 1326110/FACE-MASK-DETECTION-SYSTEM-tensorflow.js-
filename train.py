# train.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import cv2

INIT_LR = 1e-4
EPOCHS = 20
BS = 32

print("[INFO] Loading datasets from LabelMe annotations...")
DIRECTORY = "dataset"
CATEGORIES = ["with_mask", "mask_incorrectly", "no_mask"]

data = []
labels = []

if not os.path.exists(DIRECTORY):
    print(f"[ERROR] Directory '{DIRECTORY}' does not exist.")
    exit()

for filename in os.listdir(DIRECTORY):
    if filename.endswith(".json"):
        json_path = os.path.join(DIRECTORY, filename)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                annotation = json.load(f)
            
            img_filename = annotation.get("imagePath", filename.replace(".json", ".jpg"))
            img_path = os.path.join(DIRECTORY, img_filename)
            
            if not os.path.exists(img_path):
                img_path = os.path.join(DIRECTORY, filename.replace(".json", ".png"))
                if not os.path.exists(img_path):
                    img_path = os.path.join(DIRECTORY, filename.replace(".json", ".jpeg"))
                    if not os.path.exists(img_path):
                        print(f"[WARNING] Could not find the image for {json_path}")
                        continue
            
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            (h, w) = image.shape[:2]
            
            for shape in annotation.get("shapes", []):
                label = shape.get("label")
                if label not in CATEGORIES:
                    continue
                
                points = shape.get("points")
                if not points or len(points) < 2:
                    continue
                
                pt1, pt2 = points[0], points[1]
                x1, y1 = int(pt1[0]), int(pt1[1])
                x2, y2 = int(pt2[0]), int(pt2[1])
                
                startX, endX = min(x1, x2), max(x1, x2)
                startY, endY = min(y1, y2), max(y1, y2)
                
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w - 1, endX), min(h - 1, endY)
                
                face = image[startY:endY, startX:endX]
                if face.size == 0:
                    continue
                
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                
                data.append(face)
                labels.append(label)
                
        except Exception as e:
            print(f"Error processing {json_path}: {e}")

if len(data) == 0:
    print("[ERROR] No bounding box crops extracted. Please ensure LabelMe annotation files are in 'dataset' folder.")
    exit()

data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

print("[INFO] Compiling model...")
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

for layer in baseModel.layers:
    layer.trainable = False

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] Training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS
)

print("[INFO] Evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

print("[INFO] Saving mask detector model...")
model.save("mask_detector.h5", save_format="h5")

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

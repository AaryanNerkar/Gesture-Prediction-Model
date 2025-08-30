# Store all in a single dataset(csv)
# Train the ModelUse:RandomForestClassifier 
#Split data:80% for training20% for testing:Check accuracy using accuracy_score
#Save the Model  Use joblib to save the trained model in models/gesture_model.pkl.

# scikit-learn → for training/testing ML models
# joblib → for saving the trained model to a .pkl file
# pandas → for reading your CSVs

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# === SETTINGS ===
DATA_DIR = "C:/Users/Hp/OneDrive/Desktop/ADV_PY_1/data"
MODEL_DIR = "C:/Users/Hp/OneDrive/Desktop/ADV_PY_1/models"

os.makedirs(MODEL_DIR, exist_ok=True)

# === READ AND COMBINE CSV DATA SAFELY ===
all_data = []
all_labels = []

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".csv"):
        print(f"🔍 Checking file: {filename}")  # ADD THIS LINE
        label = filename.replace(".csv", "")
        filepath = os.path.join(DATA_DIR, filename)
        try:
            df = pd.read_csv(filepath)
            if df.empty:
                print(f"⚠️ Skipping empty file: {filename}")
                continue
            all_data.append(df)
            all_labels += [label] * len(df)
            print(f"✅ Loaded {filename} with {len(df)} samples.")
        except pd.errors.EmptyDataError:
            print(f"❌ ERROR: {filename} is empty or corrupt. Skipping.")
            continue

# === FINAL DATASET ===
X = pd.concat(all_data).values  # Features (63 per row)
y = all_labels                  # Labels (A, B, C, etc.)

# === TRAIN-TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === TRAIN MODEL ===
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# === EVALUATE ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {accuracy * 100:.2f}%")

# === SAVE MODEL ===
model_path = "C:/Users/Hp/OneDrive/Desktop/ADV_PY_1/models/gesture_model.pkl"

joblib.dump(model, model_path)
print(f"📁 Model saved to: {model_path}")
print("Classes:", model.classes_)


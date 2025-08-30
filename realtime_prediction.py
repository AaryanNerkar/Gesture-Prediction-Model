import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import os

# === Load trained model ===
model_path = "C:/Users/Hp/OneDrive/Desktop/ADV_PY_1/models/gesture_model.pkl"
if not os.path.exists(model_path):
    print("❌ Model file not found!")
    exit()
model = joblib.load(model_path)

# === Init MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# === Init TTS (optional) ===
engine = pyttsx3.init()
spoken_labels = set()

# === Start webcam ===
cap = cv2.VideoCapture(0)

print("🖐️ Showing hand gestures... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks += [lm.x, lm.y, lm.z]

            if len(landmarks) == 63:
                input_data = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(input_data)[0]
                prediction_text = prediction

                # Speak only if new
                if prediction not in spoken_labels:
                    engine.say(prediction)
                    engine.runAndWait()
                    spoken_labels.add(prediction)

    # Show prediction on screen
    cv2.putText(frame, f"Prediction: {prediction_text}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import csv
import os

# === SETUP ===
label = "A"  # Change this for each gesture manually for now
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
csv_path = os.path.join(data_dir, f"{label}.csv")

# Open the CSV file for writing
csv_file = open(csv_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)

# Write header
header = []
for i in range(21):
    header += [f"x{i}", f"y{i}", f"z{i}"]
csv_writer.writerow(header)

# === INIT MEDIAPIPE + CAMERA ===
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

print("Press 's' to save, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Gesture: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Collecting Data", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                row = []
                for lm in hand_landmarks.landmark:
                    row += [round(lm.x, 5), round(lm.y, 5), round(lm.z, 5)]
                csv_writer.writerow(row)
                print("✅ Sample saved:")
        else:
            print("⚠️ No hand detected, nothing saved.")

    elif key == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("asl_landmarks_model.keras")

# Define class labels (A-Z)
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Mediapipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip image for natural view
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get handedness info ("Left"/"Right")
            handedness = results.multi_handedness[hand_idx].classification[0].label

            # Extract 21 (x,y,z) landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                x, y, z = lm.x, lm.y, lm.z

                # Normalize: if Right hand → mirror x across vertical axis
                if handedness == "Right":
                    x = 1 - x  

                landmarks.extend([x, y, z])

            # Convert to numpy, reshape (1, 63)
            X = np.array(landmarks).reshape(1, -1)

            # Predict class
            pred = model.predict(X)
            class_id = np.argmax(pred)
            pred_letter = letters[class_id]

            # Display handedness + prediction
            cv2.putText(frame, f"{handedness} Hand: {pred_letter}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    cv2.imshow("ASL Live Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
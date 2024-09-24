import cv2
import pyautogui
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# Check to see if the camera opens properly or not
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# Importing hand solutions to detect hand landmarks
mp_hands = mp.solutions.hands

# Confidence detection 50%
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

last_pressed_time = 0
cooldown = 0.1  # Minimum time interval between actions
previous_hand_opened = True  # To check if the hand was opened or not

while True:
    ret, frame = cap.read()  # Captures frame one by one
    if not ret:  # Checks if the frames were captured or not
        break  # Exits the loop if frames were not successfully captured

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Changing color format so mediapipe can read it
    result = hands.process(image_rgb)  # Processes the frames to detect the hand landmarks 

    # If hand is detected will contain landmark data for the hand
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # Visualizes the hand landmarks
            print('Hand landmarks detected')

            # Extracting hand landmarks coordinates
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Checking if the hand is closed
            is_hand_closed = (
                    index_finger_tip.y > thumb_tip.y and
                    middle_finger_tip.y > thumb_tip.y and
                    ring_finger_tip.y > thumb_tip.y and
                    pinky_tip.y > thumb_tip.y
            )

            current_time = time.time()

            # Checks the previous hand position, the cooldown time so the space bar is not spammed
            if is_hand_closed and not previous_hand_opened and (current_time - last_pressed_time) > cooldown:
                pyautogui.press('space')  # Press space when hand changes from open to close
                print('Space')
                last_pressed_time = current_time

            # Update the previous state of the hand
            previous_hand_opened = is_hand_closed

    cv2.imshow('Handtroller', frame)

    # Ends the script
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyWindow('Handtroller')

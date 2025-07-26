import cv2
import numpy as np
import mediapipe as mp
import time

# Spoon calorie config
SPOON_LENGTH = 53.4  # mm
SPOON_DEPTH = 11.9   # mm
CALORIES_PER_MM3 = 0.001

def spoon_volume(length, depth):
    R = (length**2 + depth**2) / (2 * depth)
    h = depth
    return (1/3) * np.pi * h**2 * (3*R - h)

SPOON_VOLUME = spoon_volume(SPOON_LENGTH, SPOON_DEPTH)

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# HSV range for orange spoon
lower_orange = np.array([10, 100, 100])
upper_orange = np.array([25, 255, 255])

# Mouth open logic
def is_mouth_open(landmarks, image_shape):
    ih = image_shape[0]
    upper_lip = landmarks.landmark[13].y * ih
    lower_lip = landmarks.landmark[14].y * ih
    return (lower_lip - upper_lip) > 15  # threshold in pixels

# Spoon inside mouth check
def is_spoon_in_mouth(spoon_box, mouth_box):
    sx, sy, sw, sh = spoon_box
    mx, my, mw, mh = mouth_box
    center = (sx + sw/2, sy + sh/2)
    return (mx < center[0] < mx + mw) and (my < center[1] < my + mh)

cap = cv2.VideoCapture(0)

spoon_count = 0
spoon_in_mouth = False
last_spoon_time = 0
COOLDOWN = 2  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ih, iw = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    mouth_open = False
    mouth_box = (0, 0, 0, 0)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        mouth_open = is_mouth_open(landmarks, frame.shape)

        # Create mouth bounding box from landmarks [61, 291, 78, 308]
        points = [landmarks.landmark[i] for i in [61, 291, 78, 308]]
        xs = [int(p.x * iw) for p in points]
        ys = [int(p.y * ih) for p in points]
        mx, my = min(xs), min(ys)
        mw, mh = max(xs) - mx, max(ys) - my
        mouth_box = (mx, my, mw, mh)
        cv2.rectangle(frame, (mx, my), (mx+mw, my+mh), (255, 0, 0), 2)
        cv2.putText(frame, f"Mouth {'Open' if mouth_open else 'Closed'}", (mx, my-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    spoon_inside = False
    if contours:
        # Assume largest orange area is spoon
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 300:  # area threshold
            x, y, w, h = cv2.boundingRect(largest)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            spoon_inside = is_spoon_in_mouth((x, y, w, h), mouth_box)
            if spoon_inside:
                cv2.putText(frame, "Spoon in mouth", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    current_time = time.time()
    if spoon_inside and mouth_open:
        spoon_in_mouth = True

    if spoon_in_mouth and not mouth_open and (current_time - last_spoon_time) > COOLDOWN:
        spoon_count += 1
        spoon_in_mouth = False
        last_spoon_time = current_time

    total_calories = spoon_count * SPOON_VOLUME * CALORIES_PER_MM3

    # UI
    cv2.putText(frame, f"Spoons: {spoon_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Calories: {total_calories:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Orange Spoon Calorie Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

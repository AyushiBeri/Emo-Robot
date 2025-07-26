import cv2
import subprocess
import numpy as np
import mediapipe as mp
import threading
import time
import pygame

# ---- Initialize Pygame mixer for beep ----
pygame.mixer.init(frequency=44100, size=-16, channels=1)

def play_beep():
    frequency = 880  # Hz
    duration = 100   # milliseconds
    sample_rate = 44100
    n_samples = int(sample_rate * duration / 1000)
    wave = (4096 * np.sin(2.0 * np.pi * np.arange(n_samples) * frequency / sample_rate)).astype(np.int16)
    sound = pygame.sndarray.make_sound(wave)
    sound.play()

# ---- Constants ----
WIDTH, HEIGHT = 640, 480
BASELINE = 0.06  # meters

frames = [None, None]
robot_state = "Stopped"

# ---- MediaPipe Setup ----
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=50, detectShadows=False)

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,
    blockSize=5,
    P1=8 * 3 * 5 ** 2,
    P2=32 * 3 * 5 ** 2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# ---- Stream from Camera ----
def stream_camera(command, idx):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)
    while True:
        raw = proc.stdout.read(WIDTH * HEIGHT * 3 // 2)
        if not raw:
            break
        yuv = np.frombuffer(raw, dtype=np.uint8).reshape((int(HEIGHT * 1.5), WIDTH))
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        frames[idx] = frame
    proc.terminate()

def compute_depth(left_frame, right_frame, x, y):
    gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
    disparity_map = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    if y >= disparity_map.shape[0] or x >= disparity_map.shape[1]:
        return None
    disparity = disparity_map[y, x]
    if disparity > 0:
        focal_length_px = (3.6 / 3.984) * WIDTH
        depth = (focal_length_px * BASELINE) / disparity
        return depth * 100  # in cm
    return None

def is_mouth_open(landmarks, img_h, img_w, threshold=15):
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]
    top = int(upper_lip.y * img_h)
    bottom = int(lower_lip.y * img_h)
    distance = abs(bottom - top)
    return distance > threshold

def is_looking_forward(landmarks, img_w, eye_closed_threshold=5):
    # Check for enough landmarks first
    if len(landmarks) < 474:
        return False

    # Calculate horizontal iris position for left eye
    left_outer = landmarks[33].x * img_w
    left_inner = landmarks[133].x * img_w
    left_iris = landmarks[468].x * img_w
    left_center = (left_outer + left_inner) / 2
    left_diff = abs(left_iris - left_center)

    # Calculate horizontal iris position for right eye
    right_outer = landmarks[362].x * img_w
    right_inner = landmarks[263].x * img_w
    right_iris = landmarks[473].x * img_w
    right_center = (right_outer + right_inner) / 2
    right_diff = abs(right_iris - right_center)

    # Check vertical eye openness (using upper and lower eyelid landmarks)
    # Left eye vertical distance
    left_eye_top = landmarks[159].y * img_w
    left_eye_bottom = landmarks[145].y * img_w
    left_eye_height = abs(left_eye_bottom - left_eye_top)

    # Right eye vertical distance
    right_eye_top = landmarks[386].y * img_w
    right_eye_bottom = landmarks[374].y * img_w
    right_eye_height = abs(right_eye_bottom - right_eye_top)

    eyes_open = (left_eye_height > eye_closed_threshold) and (right_eye_height > eye_closed_threshold)

    # Return True only if eyes are open and iris close to center for both eyes
    return eyes_open and (left_diff < 8) and (right_diff < 8)

def control_robot_arm(enable):
    global robot_state
    if enable and robot_state == "Stopped":
        robot_state = "Feeding… 🤖🍽️"
        print("[Robot] Feeding gesture detected - arm moving")
    elif not enable and robot_state == "Feeding… 🤖🍽️":
        robot_state = "Stopped"
        print("[Robot] Stopped")

# ---- Camera commands ----
cmd_left = [
    "libcamera-vid", "--camera", "0", "-t", "0",
    "--width", str(WIDTH), "--height", str(HEIGHT),
    "--codec", "yuv420", "-o", "-"
]
cmd_right = [
    "libcamera-vid", "--camera", "1", "-t", "0",
    "--width", str(WIDTH), "--height", str(HEIGHT),
    "--codec", "yuv420", "-o", "-"
]

t_left = threading.Thread(target=stream_camera, args=(cmd_left, 0))
t_right = threading.Thread(target=stream_camera, args=(cmd_right, 1))
t_left.daemon = True
t_right.daemon = True
t_left.start()
t_right.start()

print("Starting main loop. Press 'q' to quit...")

MOTION_CONTOUR_AREA_THRESHOLD = 1500

def put_retro_text(img, text, pos, color):
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    scale = 0.7
    thickness = 1
    shadow_color = (20, 20, 20)  # Dark shadow
    x, y = pos
    # Shadow
    cv2.putText(img, text, (x+1, y+1), font, scale, shadow_color, thickness+1, lineType=cv2.LINE_AA)
    # Main text
    cv2.putText(img, text, (x, y), font, scale, color, thickness, lineType=cv2.LINE_AA)

try:
    while True:
        if frames[0] is None or frames[1] is None:
            time.sleep(0.01)
            continue

        left_frame = frames[0].copy()
        right_frame = frames[1].copy()

        rgb_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
        h, w, _ = left_frame.shape

        face_results = face_mesh.process(rgb_left)
        hand_results = hands.process(rgb_left)

        fg_mask = bg_subtractor.apply(left_frame)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=1)
        thresh = cv2.dilate(thresh, None, iterations=2)

        mouth_rect = None
        obstacle_detected = False

        if face_results.multi_face_landmarks:
            face = face_results.multi_face_landmarks[0]
            mp_drawing.draw_landmarks(left_frame, face, mp_face_mesh.FACEMESH_TESSELATION)

            x_mouth = int((face.landmark[13].x + face.landmark[14].x) / 2 * w)
            y_mouth = int((face.landmark[13].y + face.landmark[14].y) / 2 * h)

            roi_w, roi_h = 140, 120
            x1, y1 = max(0, x_mouth - roi_w // 2), max(0, y_mouth - roi_h // 2)
            x2, y2 = min(w, x1 + roi_w), min(h, y1 + roi_h)
            mouth_rect = (x1, y1, x2, y2)

            cv2.rectangle(left_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(left_frame, "Mouth", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            mouth_open = is_mouth_open(face.landmark, h, w)
            looking_forward = is_looking_forward(face.landmark, w)

            nose = face.landmark[1]
            x_nose = int(nose.x * w)
            y_nose = int(nose.y * h)
            depth = compute_depth(left_frame, right_frame, x_nose, y_nose)
            if depth:
                cv2.putText(left_frame, f"Depth: {depth:.2f} cm", (x_nose, y_nose - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            mouth_open = False
            looking_forward = False

        if hand_results.multi_hand_landmarks and mouth_rect:
            x1, y1, x2, y2 = mouth_rect
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(left_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    x_hand = int(lm.x * w)
                    y_hand = int(lm.y * h)
                    if x1 < x_hand < x2 and y1 < y_hand < y2:
                        obstacle_detected = True
                        break
                if obstacle_detected:
                    break

        if mouth_rect and not obstacle_detected:
            x1, y1, x2, y2 = mouth_rect
            roi_motion = thresh[y1:y2, x1:x2]
            contours, _ = cv2.findContours(roi_motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > MOTION_CONTOUR_AREA_THRESHOLD:
                    obstacle_detected = True
                    x_m, y_m, w_box, h_box = cv2.boundingRect(cnt)
                    cv2.rectangle(left_frame, (x1 + x_m, y1 + y_m), (x1 + x_m + w_box, y1 + y_m + h_box), (0, 0, 255), 2)
                    break

        if obstacle_detected:
            play_beep()

        arm_enable = mouth_open and looking_forward and not obstacle_detected
        control_robot_arm(arm_enable)

        # ---- UI STATUS OVERLAYS ----

        # Top-left: Obstacle - use retro font
        obstacle_status = "Obstacle: Detected" if obstacle_detected else "Obstacle: Clear"
        put_retro_text(left_frame, obstacle_status, (10, 30), (0, 0, 255) if obstacle_detected else (0, 255, 0))

        # Bottom-left: Robot state - use retro font
        put_retro_text(left_frame, f"Robot State: {robot_state}", (10, h - 20), (255, 255, 255))

        # Top-right: Mouth and Eye status in small retro font with shadow
        retro_x = w - 180
        retro_y = 30
        put_retro_text(left_frame, f"Mouth: {'Open' if mouth_open else 'Closed'}", (retro_x, retro_y), (0, 255, 255))
        put_retro_text(left_frame, f"Eyes: {'Forward' if looking_forward else 'Not Forward'}", (retro_x, retro_y + 25), (0, 255, 255))

        cv2.imshow("Left Frame", left_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cv2.destroyAllWindows()

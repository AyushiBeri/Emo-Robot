import cv2
import mediapipe as mp
import numpy as np
import time


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
LIPS = mp_face_mesh.FACEMESH_LIPS
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]


emotions = {
    "happy": { "color": (0, 255, 0)},
    "sad": {"color": (255, 0, 0)},
    "angry": {"color": (0, 0, 255)},
    "surprise": {"color": (0, 255, 255)},
    "neutral": {"color": (255, 255, 255)},
    "fear": {"color": (255, 140, 0)},
    "disgust": {"color": (138, 43, 226)}
}

def distance(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

def get_emotion(landmarks):
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]
    right_eye_top = landmarks[386]
    right_eye_bottom = landmarks[374]
    iris_left = landmarks[468]

    face_width = distance(landmarks[234], landmarks[454])

    mouth_open = distance(top_lip, bottom_lip) / face_width
    mouth_stretch = distance(left_mouth, right_mouth) / face_width
    eye_open = (distance(left_eye_top, left_eye_bottom) + distance(right_eye_top, right_eye_bottom)) / (2 * face_width)

    eye_top_avg = (left_eye_top.y + right_eye_top.y) / 2
    eye_bottom_avg = (left_eye_bottom.y + right_eye_bottom.y) / 2
    iris_avg = iris_left.y
    eye_center_y = (eye_top_avg + eye_bottom_avg) / 2
    sad_offset = iris_avg - eye_center_y

    print(f"[DEBUG] mouth_open={mouth_open:.3f}, mouth_stretch={mouth_stretch:.3f}, eye_open={eye_open:.3f}, sad_offset={sad_offset:.3f}")

    if mouth_stretch > 0.40 and mouth_open < 0.06:
        return "happy"
    elif mouth_open >= 0.12:
        return "surprise"
    elif 0.06 < mouth_open < 0.12:
        return "fear"
    elif sad_offset > 0.01 and eye_open < 0.04:
        return "sad"
    elif mouth_open < 0.03 and eye_open < 0.08 and mouth_stretch < 0.38:
        return "disgust"
    elif eye_open > 0.096 and mouth_open < 0.06:
        return "angry"
    else:
        return "neutral"


def get_latest_expression():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "neutral"

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmark_list = face_landmarks.landmark
            emotion = get_emotion(landmark_list)
            return emotion

    return "neutral"

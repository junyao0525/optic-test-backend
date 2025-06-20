from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np
import os
import joblib
import tempfile
from datetime import datetime
from scipy.spatial import distance as dist

router = APIRouter(prefix="/fatigue", tags=["Fatigue Analysis"])

# Load model and scaler using absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(current_dir, "mlp_model.pkl"))
scaler = joblib.load(os.path.join(current_dir, "mlp_scaler.pkl"))
class_names = ["High", "Normal"]

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C != 0 else 0

def calculate_perclos(ear_list, threshold=0.20):
    closed = sum(ear < threshold for ear in ear_list)
    return closed / len(ear_list) if ear_list else 0

@router.post("/analyze")
async def analyze_fatigue(file: UploadFile = File(...)):
    contents = await file.read()

    # Save to temporary file for OpenCV processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(contents)
        tmp_path = tmp_file.name

    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True
    )

    EARs, blink_durations = [], []
    blink_count = 0
    blink_flag = False
    blink_start = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]
            left_eye = np.array([landmarks[i] for i in LEFT_EYE])
            right_eye = np.array([landmarks[i] for i in RIGHT_EYE])
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            EARs.append(ear)

            if ear < 0.2 and not blink_flag:
                blink_flag = True
                blink_start = frame_count
            elif ear >= 0.2 and blink_flag:
                blink_flag = False
                if blink_start is not None:
                    blink_duration = frame_count - blink_start
                    blink_durations.append(blink_duration)
                    blink_count += 1
                    blink_start = None

    cap.release()
    face_mesh.close()
    os.remove(tmp_path)  # Clean up temp file

    duration_sec = frame_count / fps if fps > 0 else 1

    perclos = calculate_perclos(EARs)
    ear_mean = np.mean(EARs) if EARs else 0
    ear_std = np.std(EARs) if EARs else 0
    blink_rate = (blink_count / duration_sec) * 60
    avg_blink_dur = (np.mean(blink_durations) / fps) if blink_durations else 0

    input_features = np.array([[perclos, avg_blink_dur, ear_std, ear_mean, blink_rate]])
    scaled_input = scaler.transform(input_features)
    prediction = model.predict(scaled_input)[0]

    return JSONResponse({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "PERCLOS": round(perclos, 4),
        "Avg Blink Duration": round(avg_blink_dur, 4),
        "EAR Mean": round(ear_mean, 4),
        "EAR Std": round(ear_std, 4),
        "Blink Rate": round(blink_rate, 2),
        "Predicted Fatigue Class": class_names[prediction]
    })

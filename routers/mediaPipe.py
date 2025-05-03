import os
import io
from datetime import datetime
import math
from fastapi import APIRouter, UploadFile, File
import numpy as np
import cv2
import mediapipe as mp
import json

router = APIRouter(prefix="/mediapipe", tags=["face-recognition-mediapipe"])

UPLOAD_FOLDER = "uploads/mediapipe/input"
OUTPUT_FOLDER = "uploads/mediapipe/output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Constants
KNOWN_FACE_WIDTH_CM = 14  # Average adult face width
FOCAL_LENGTH_PX = 1100     # Approximate focal length, needs calibration for your camera


def measure_distance(pupil_distance_px):
    return (KNOWN_FACE_WIDTH_CM * FOCAL_LENGTH_PX) / pupil_distance_px


@router.post("/detect-face/")
async def detect_faces_mediapipe(file: UploadFile = File(...)):
    try:
        if not file:
            return {"error": "No file uploaded.", "message": "Please provide an image."}

        contents = await file.read()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        with open(file_path, "wb") as f:
            f.write(contents)

        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"Image shape: {img.shape}")  

        # Draw center area box
        h, w, _ = img.shape
        center_width = w // 3
        center_height = h // 3
        frame_center_x = w // 2
        frame_center_y = h // 2
        
        # Calculate the coordinates for center area box
        center_x1 = frame_center_x - center_width // 2
        center_y1 = frame_center_y - center_height // 2
        center_x2 = frame_center_x + center_width // 2
        center_y2 = frame_center_y + center_height // 2
        
        # Draw center area box in black with 2px thickness
        cv2.rectangle(img, (center_x1, center_y1), (center_x2, center_y2), (0, 0, 0), 2)

        results = face_mesh.process(img_rgb)

        face_data = [] 
        landmark_data = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                # Use pupil landmarks (left: 468, right: 473 in MediaPipe's 478 landmarks)
                left_pupil = face_landmarks.landmark[468]  # Left pupil center
                right_pupil = face_landmarks.landmark[473] # Right pupil center

                # Convert to pixel coordinates
                x1, y1 = int(left_pupil.x * w), int(left_pupil.y * h)
                x2, y2 = int(right_pupil.x * w), int(right_pupil.y * h)

                # Euclidean distance
                pixel_distance = math.hypot(x2 - x1, y2 - y1)
                
                # Calculate distance to screen
                distance_cm = measure_distance(pixel_distance)
                
                # Draw distance info on the image
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.putText(img, f"{distance_cm:.1f} cm", (mid_x, mid_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Determine if the face is centered
                is_centered = (
                    center_x1 < mid_x < center_x2 and
                    center_y1 < mid_y < center_y2
                )

                # Determine if face is too small (means too far)
                # when small then 400
                min_face_pixel_distance = w // 6 
                is_too_far = pixel_distance < min_face_pixel_distance

                is_too_near = pixel_distance > min_face_pixel_distance * 2

                # Add to results
                face_info = {
                    "distance_cm": round(distance_cm),
                    "pixel_distance": round(pixel_distance),
                    "is_too_near": is_too_near,
                    "is_centered": is_centered,
                    "is_too_far": is_too_far,
                }
                face_data.append(face_info)

                print(f"Face Info: {face_info}")

                # Store all landmarks for future use if needed
                single_landmarks = [
                    {"x": int(lm.x * w), "y": int(lm.y * h)}
                    for lm in face_landmarks.landmark
                ]
                landmark_data.append(single_landmarks)

        # Save annotated image
        annotated_filename = f"annotated_{timestamp}_{file.filename}"
        annotated_path = os.path.join(OUTPUT_FOLDER, annotated_filename)
        cv2.imwrite(annotated_path, img)
        import json

        # Build JSON data
        record = {
            "timestamp": timestamp,
            "original_image": file_path,
            "annotated_image": annotated_path,
            "face_count": len(face_data),
            "faces": face_data,
        }

        # Optional: Save to .json file
        json_output_path = os.path.join(OUTPUT_FOLDER, f"record_{timestamp}.json")
        with open(json_output_path, "w") as f:
            json.dump(record, f, indent=2)
        
        result ={
            "timestamp": timestamp,
            "face_count": len(face_data),
            "faces": face_data,
        }

        return result
    

    except Exception as e:
        return {"error": str(e), "message": "An error occurred while processing the image."}

@router.get("/mediapipe-test/")
async def hello_world_mediapipe():
    return {"data": "Testing mediapipe router operational."}
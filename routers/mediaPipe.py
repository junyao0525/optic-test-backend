import os
import io
from datetime import datetime
import math
from fastapi import APIRouter, UploadFile, File
from logger.logger import logger
import numpy as np
import cv2
import mediapipe as mp
import json
from PIL import Image
from PIL.ExifTags import TAGS

router = APIRouter(prefix="/mediapipe", tags=["face-recognition-mediapipe"])

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Constants
KNOWN_PUPIL_WIDTH_MM = 63     # Average inter-pupillary distance for adults
DEFAULT_FOCAL_LENGTH_PX = 2500 # Increased from 2000 to 2500 for better calibration
DEFAULT_FOV_DEGREES = 60      # Default field of view in degrees
MIN_PUPIL_DISTANCE_PX = 50    # Minimum pupil distance in pixels to consider valid
MAX_PUPIL_DISTANCE_PX = 500   # Maximum pupil distance in pixels to consider valid
CALIBRATION_FACTOR = 1.3      # Calibration factor to adjust for systematic error

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

        max_size = 10 * 1024 * 1024  # 10 MB
        if len(contents) > max_size:
            return {"error": "File too large.", "message": "The uploaded file exceeds the maximum size of 10 MB."}

        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"Image shape: {img.shape}")
        logger.info(f"Image shape: {img.shape}")

        # # Draw center area box
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

                # Store all landmarks for future use if needed
                single_landmarks = [
                    {"x": int(lm.x * w), "y": int(lm.y * h)}
                    for lm in face_landmarks.landmark
                ]
                landmark_data.append(single_landmarks)

        
        result ={
            "face_count": len(face_data),
            "faces": face_data,
        }
        print(f"Face detection result: {result}")
        logger.info(f"Face detection result: {result}")

        return result
    

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return {"error": str(e), "message": "An error occurred while processing the image."}

@router.get("/mediapipe-test/")
async def hello_world_mediapipe():
    return {"data": "Testing mediapipe router operational"}
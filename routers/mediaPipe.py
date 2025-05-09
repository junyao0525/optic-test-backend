import os
import io
from datetime import datetime
import math
from fastapi import APIRouter, UploadFile, File
import numpy as np
import cv2
import mediapipe as mp
import json
from PIL import Image
from PIL.ExifTags import TAGS

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

def extract_exif_metadata(image):
    """Extracts and returns EXIF metadata as a dictionary."""
    exif_data = image._getexif()
    if not exif_data:
        return {}

    metadata = {}
    for tag_id, value in exif_data.items():
        tag = TAGS.get(tag_id, tag_id)
        metadata[tag] = value
    return metadata

def calculate_fov_from_focal_length(focal_length_mm, sensor_width_mm):
    """Calculate field of view in degrees from focal length and sensor width."""
    return 2 * math.atan(sensor_width_mm / (2 * focal_length_mm)) * (180 / math.pi)

def calibrate_camera(known_distance_cm, pupil_distance_px, image_width_px):
    """Calibrate camera parameters using a known distance measurement."""
    focal_length_px = (pupil_distance_px * known_distance_cm * 10) / KNOWN_PUPIL_WIDTH_MM
    return focal_length_px

def measure_distance(image_bytes, pupil_distance_px):
    """
    Calculate distance to face using improved formula that considers FOV and calibration.
    
    Args:
        image_bytes: Raw image bytes
        pupil_distance_px: Distance between pupils in pixels
        
    Returns:
        float: Distance in centimeters, or None if calculation failed
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        metadata = extract_exif_metadata(image)
        image_width_px = metadata.get("ExifImageWidth") or image.width

        # Try to get camera parameters from EXIF
        focal_length_mm = metadata.get("FocalLength")
        focal_length_35mm = metadata.get("FocalLengthIn35mmFilm")

        if focal_length_mm and focal_length_35mm:
            # Convert focal length from tuple if necessary
            if isinstance(focal_length_mm, tuple):
                focal_length_mm = focal_length_mm[0] / focal_length_mm[1]
            
            # Calculate sensor parameters
            crop_factor = focal_length_35mm / focal_length_mm
            sensor_width_mm = 36 / crop_factor  # Full-frame width is 36mm
            
            # Calculate FOV
            fov_degrees = calculate_fov_from_focal_length(focal_length_mm, sensor_width_mm)
            
            # Convert focal length to pixels
            focal_length_px = focal_length_mm * (image_width_px / sensor_width_mm)
            
            # Calculate distance using improved formula that considers FOV
            # Formula: distance = (known_width * focal_length) / (pixel_width * tan(FOV/2))
            fov_rad = math.radians(fov_degrees)
            distance_mm = (KNOWN_PUPIL_WIDTH_MM * focal_length_px) / (pupil_distance_px * math.tan(fov_rad/2))
            
            # Apply calibration factor and convert to cm
            distance_cm = (distance_mm / 10) * CALIBRATION_FACTOR
            
            return max(0, distance_cm)  # Ensure non-negative distance
            
        else:
            print("Incomplete EXIF data. Using calibrated fallback.")
            # Use calibrated focal length if available, otherwise use default
            focal_length_px = DEFAULT_FOCAL_LENGTH_PX
            
            # Calculate distance using simplified formula with calibration factor
            distance_cm = (KNOWN_PUPIL_WIDTH_MM * focal_length_px) / (pupil_distance_px * 10) * CALIBRATION_FACTOR
            
            return max(0, distance_cm)

    except Exception as e:
        print(f"Error calculating distance: {e}")
        return None

@router.post("/detect-face/")
async def detect_faces_mediapipe(file: UploadFile = File(...)):
    try:
        if not file:
            return {"error": "No file uploaded.", "message": "Please provide an image."}

        contents = await file.read()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"Image shape: {img.shape}")  

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
                distance_cm = measure_distance(contents,pixel_distance)
                
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
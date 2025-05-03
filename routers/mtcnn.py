# routers/mtcnn.py
import math
from fastapi import APIRouter, UploadFile, File
from mtcnn import MTCNN
from PIL import Image
import numpy as np
import io
import os
from datetime import datetime


router = APIRouter(prefix="/mtcnn", tags=["face-recognition"])
detector = MTCNN()
REAL_EYE_DISTANCE = 6.3  # Average human interpupillary distance
FOCAL_LENGTH = 700  # Focal length of the camera (in pixels)

# Function to calculate distance from eye keypoints
def calculate_distance(left_eye, right_eye, focal_length, real_eye_distance):
    pixel_distance = math.sqrt((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2)
    distance = (focal_length * real_eye_distance) / pixel_distance
    return distance

# 1. face > 2
# 2. distance < 100cm
# 3. eye distance < 10cm


UPLOAD_FOLDER = "uploads"

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@router.post("/detect-face/")
async def detect_face(file: UploadFile = File(...)):
    try:
        if not file:
            print("No file uploaded.")
            return {
                "error": "No file uploaded.",
                "message": "Please provide an image file.",
            }

        print(f"Received file: {file.filename}")
        contents = await file.read()

        # Save the uploaded file (optional)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(file_path, "wb") as f:
            f.write(contents)

        # Process the image
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)  # Convert to numpy array
        
        print(f"Image shape: {image_np.shape}")  # Should be (height, width, 3)
        
        # Adjust MTCNN parameters for better detection
        detector = MTCNN(
            # min_face_size=20,  # Smaller minimum face size
            # scale_factor=0.709  # Smaller scale factor for better detection
        )
        
        result = detector.detect_faces(image_np)
        print(f"Detection results: {result}")
        
        face_count = len(result)
        
        if face_count == 0:
            return {
                "error": "No faces detected",
                "message": "Try adjusting camera distance or lighting",
                "image_size": f"{image_np.shape[1]}x{image_np.shape[0]}",
                "results": result,
            }
            
        if face_count > 2:
            return {
                "error": "Too many faces detected",
                "face_count": face_count,
            }

        distances = []
        for face in result:
            keypoints = face.get("keypoints", {})
            if "left_eye" in keypoints and "right_eye" in keypoints:
                left_eye = keypoints["left_eye"]
                right_eye = keypoints["right_eye"]
                distance_cm = calculate_distance(
                    left_eye, right_eye, 
                    FOCAL_LENGTH, REAL_EYE_DISTANCE
                )
                distances.append(distance_cm)

        final_result ={
            "face_count": face_count,
            "distances_cm": distances[0] if distances else None,
            "detection_confidence": [f["confidence"] for f in result],
            "image_size": f"{image_np.shape[1]}x{image_np.shape[0]}"
        }

        print(f"Final result: {final_result}")

        return {
          final_result
        }

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return {
            "error": str(e),
            "message": "An error occurred while processing the image.",
        }
    
# testing the router is working
@router.get("/hello-world/")
async def helloWorld():
    return {"data": "testing the router is working"}
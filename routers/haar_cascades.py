import math
from fastapi import APIRouter, UploadFile, File
from PIL import Image
import numpy as np
import io
import os
from datetime import datetime
import cv2

router = APIRouter(prefix="/haar", tags=["face-recognition"])


UPLOAD_FOLDER = "uploads/haar/input"
OUTPUT_FOLDER = "uploads/haar/output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

current_dir = os.path.dirname(os.path.realpath(__file__))
cascade_path = os.path.join(current_dir, "utils", "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

@router.post("/detect-face/")
async def detect_faces(file: UploadFile = File(...)):
    try:
        if not file:
            print("No file uploaded.")
            return {
                "error": "No file uploaded.",
                "message": "Please provide an image file.",
            }

        contents = await file.read()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(file_path, "wb") as f:
            f.write(contents)

        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        face_data = []
        for (x, y, w, h) in faces:
            face_data.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            })
            
        # Draw rectangle in green color with 2px thickness
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        # Calculate distances if faces were detected
        distances_cm = []
        if len(faces) > 0:
            # Camera parameters (these would need to be calibrated for your specific camera)
            FOCAL_LENGTH = 600  # Approximate focal length in pixels
            KNOWN_FACE_WIDTH = 16  # Average human face width in cm
            
            for (x, y, w, h) in faces:
                # Calculate distance using the formula: distance = (known_width * focal_length) / perceived_width
                distance = (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / w
                distances_cm.append(round(distance, 2))
                
                # Add distance text above the rectangle
                cv2.putText(img, f"{distance:.1f} cm", (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode the image with rectangles as JPEG
        _, buffer = cv2.imencode('.jpg', img)
        img_bytes = buffer.tobytes()
        
        # Save the annotated image
        annotated_filename = f"annotated_{timestamp}_{file.filename}"
        annotated_path = os.path.join(OUTPUT_FOLDER, annotated_filename)
        with open(annotated_path, "wb") as f:
            f.write(img_bytes)

        return {
            "face_count": len(faces),
            "faces": face_data
        }

    except Exception as e:
        return {
            "error": str(e),
            "message": "An error occurred while processing the image."
        }

# Testing the router
@router.get("/haar-test/")
async def helloWorld():
    return {"data": "testing haar-test router is working"}

from fastapi import APIRouter, File, UploadFile


router = APIRouter(prefix="/whisper-lora", tags=["audio-transcriber"])

@router.post("/audio-transcriber/")
async def audio_transcriber(file: UploadFile = File(...)):
    try:
        if not file:
            return {"error": "No file uploaded.", "message": "Please provide an image."}
        
        return UploadFile
    except Exception as e:
        return {"error": str(e), "message": "An error occurred while processing the image."}
    


@router.get("/lora-test/")
async def hello_world_mediapipe():
    return {"data": "Testing whisper router operational"}
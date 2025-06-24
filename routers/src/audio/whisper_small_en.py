import os
from pathlib import Path
import torch
from fastapi import APIRouter, Form, UploadFile, File, logger
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pydub import AudioSegment
import tempfile
import numpy as np
from peft import PeftModel, PeftConfig

router = APIRouter(prefix="/whisper-small", tags=["audio-transcriber"])

# Resolve model path
current_dir = Path(__file__).resolve().parent
peft_model_dir = (current_dir / ".." / ".." / "utils" / "whisper-en-small").resolve()

# Load base + LoRA model
peft_config = PeftConfig.from_pretrained(peft_model_dir)
base_model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, peft_model_dir)
model.eval()




@router.post("/audio-transcriber/")
async def audio_transcriber(file: UploadFile = File(...), language: str = Form("en")):
    try:
        if not file:
            return {"error": "No file uploaded."}
        
        # Log the incoming request
        print(f"Received file: {file.filename}, content_type: {file.content_type}, size: {file.size if hasattr(file, 'size') else 'unknown'}")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('audio/'):
            print(f"Invalid file type: {file.content_type}")
            return {"error": "Invalid file type. Only audio files are accepted."}
        
        language = "en"
        processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
        
        # 2. Save audio to temp file
        try:
            content = await file.read()
            if not content:
                print("Empty file received")
                return {"error": "Empty file received"}
            
            print(f"File content size: {len(content)} bytes")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
                print(f"Saved temporary file at: {tmp_path}")
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            return {"error": f"Error saving file: {str(e)}"}

        # 3. Preprocess audio
        try:
            audio = AudioSegment.from_file(tmp_path, format="m4a")
            audio = audio.set_channels(1).set_frame_rate(16000).normalize()
            samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
            print(f"Audio length: {len(audio)} ms, Samples: {len(samples)}")
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return {"error": f"Error processing audio: {str(e)}"}

        # 4. Tokenize
        try:
            input_features = processor(
                samples, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features
        except Exception as e:
            print(f"Error tokenizing audio: {str(e)}")
            return {"error": f"Error tokenizing audio: {str(e)}"}

        # 5. Generate transcription
        try:
            predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            print(f"Transcription: {transcription}")
            return {"transcription": transcription}
        except Exception as e:
            print(f"Error generating transcription: {str(e)}")
            return {"error": f"Error generating transcription: {str(e)}"}

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {"error": str(e), "message": "An error occurred while processing the audio."}
    finally:
        # Clean up temporary file
        try:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except Exception as e:
            print(f"Error cleaning up temporary file: {str(e)}")


@router.get("/test/")
async def test():
     return {"data": "Testing whishper router operational"}
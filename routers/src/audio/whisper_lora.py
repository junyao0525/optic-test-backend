from pathlib import Path
import torch
from fastapi import APIRouter, UploadFile, File
from peft import PeftModel, PeftConfig
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pydub import AudioSegment
import tempfile
import numpy as np

router = APIRouter(prefix="/whisper-lora", tags=["audio-transcriber"])

# Resolve model path
current_dir = Path(__file__).resolve().parent
peft_model_dir = (current_dir / ".." / ".." / "utils" / "whisper-lora").resolve()

# Load base + LoRA model
peft_config = PeftConfig.from_pretrained(peft_model_dir)
base_model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, peft_model_dir)
model.eval()

# Load processor
processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")


@router.post("/audio-transcriber/")
async def audio_transcriber(file: UploadFile = File(...)):
    try:
        if not file:
            return {"error": "No file uploaded."}

        # Save UploadFile to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Load and normalize audio with pydub
        audio = AudioSegment.from_file(tmp_path, format="mp3")
        audio = audio.set_channels(1).set_frame_rate(16000)  # mono, 16kHz
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        speech_tensor = torch.tensor(samples).unsqueeze(0)  # shape: [1, T]

        # Tokenize input
        input_features = processor(speech_tensor.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features

        # Inference
        with torch.no_grad():
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        return {"transcription": transcription}

    except Exception as e:
        return {"error": str(e), "message": "An error occurred while processing the audio."}

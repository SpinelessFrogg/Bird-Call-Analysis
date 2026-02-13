from fastapi import FastAPI, UploadFile, File
import numpy as np
from keras.models import load_model
from preprocessing.audio import decode_audiosegment
from preprocessing.pipeline import waveform_to_melspec, prepare_single
from config import MODEL_DIR
from pydub import AudioSegment
from io import BytesIO


app = FastAPI()

model_name = "2-10-26_fixedwidth_extra_conv.keras"
model = load_model(f"{MODEL_DIR}{model_name}")

CLASS_NAMES = np.load(f"{MODEL_DIR}class_names.npy")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        audio = AudioSegment.from_file(BytesIO(contents), format="mp3")
        samples, sr = decode_audiosegment(audio)
        spec = waveform_to_melspec(samples, sr)
        spec = prepare_single(spec)
    except Exception as e:
        return {"error": str(e)}

    pred = model.predict(spec)
    pred_class = int(np.argmax(pred))
    confidence = float(np.max(pred))
    
    return {
        "species": CLASS_NAMES[pred_class],
        "confidence": confidence
    }
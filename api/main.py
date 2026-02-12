from fastapi import FastAPI, UploadFile, File
import numpy as np
from keras.models import load_model
import io

from preprocessing.dataset_builder import DatasetBuilder
from config import MODEL_DIR

app = FastAPI()

model_name = "2-10-26_fixedwidth_extra_conv.keras"
model = load_model(f"{MODEL_DIR}{model_name}")

CLASS_NAMES = np.load(f"{MODEL_DIR}class_names.npy")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

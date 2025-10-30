from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from pydantic import BaseModel
import os
import numpy as np
import uvicorn

app = FastAPI()

origins = [
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",
    # "https://your-frontend.netlify.app" --> later, add your deployed frontend URL here
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "asl_landmarks_model.keras")
model = load_model(MODEL_PATH)
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

class LandmarkData(BaseModel):
    landmarks: list[list[float]]
    handedness: str

@app.post("/predict")
def predict(data: LandmarkData):
    print("Received landmarks:", len(data.landmarks), flush=True)
    landmarks = []
    for (x, y, z) in data.landmarks:
        if data.handedness == "Right":
            x = 1 - x
        landmarks.extend([x, y, z])

    x = np.array(landmarks).reshape(1, -1)
    pred = model.predict(x)
    class_id = int(np.argmax(pred))
    pred_letter = letters[class_id]


    print(f"Predicted letter: {pred_letter}", flush=True)
    return {"prediction": pred_letter}

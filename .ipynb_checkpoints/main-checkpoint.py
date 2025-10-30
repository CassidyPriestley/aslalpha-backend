from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image

app = FastAPI()

origins = [
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",
    # later, add your deployed frontend URL here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # list of allowed origins
    allow_credentials=True,
    allow_methods=["*"],            # allow all HTTP methods
    allow_headers=["*"],            # allow all headers
)


class ImageRequest(BaseModel):
    # base64 encoded image str from frontend
    image: str

@app.post("/predict")
async def predict(req: ImageRequest):
    # Strip the "data:image/jpeg;base64," prefix if it exists
    image_data = req.image.split(",")[1] if "," in req.image else req.image

    # Decode the base64 string to bytes
    image_bytes = base64.b64decode(image_data)

    # Convert bytes to PIL image (for later ML processing)
    image = Image.open(BytesIO(image_bytes))

    # Todo: Preprocess image & run through ML model
    # Return dummy prediction
    return {"prediction": "testing...."}

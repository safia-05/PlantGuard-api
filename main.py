from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io

#Setup app
app = FastAPI(
    title="PlantGuard",
    description="Upload a plant image to identify toxic/non.",
    version="1.0.0",
)

#allow all clients to use api
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#Constants
IMG_SIZE = 260         
LOW_CONF_THRESHOLD  = 0.60
HIGH_CONF_THRESHOLD = 0.80

CLASS_NAMES = ["Toxic", "Non-Toxic"]

CLASS_INFO = {
    "Non-Toxic": {
        "color": "green",
        "advice": "This plant appears to be safe.",
    },
    "Toxic": {
        "color": "red",
        "advice": "Stay away and call emergency if touched.",
    },
    "Unknown": {
        "color": "orange",
        "advice": "Not sure — avoid touching and consult an expert.",
    },
}

#model path
MODEL_PATH = "best_efficientnet_b2.pth"

#build architecture
def build_model():
    model = models.efficientnet_b2(weights=None)
    num_features = model.classifier[1].in_features  # 1408 for EfficientNet-B2
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),          # dropout rate may vary; 0.4 is common
        nn.Linear(num_features, 2)  # direct to 2 classes
    )
    return model

#load model at startup 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = build_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print(f"Model loaded from '{MODEL_PATH}' on {device}")

#preprocessing info mn el notebook
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    ),
])

#routes
@app.get("/")
def root():
    return {"message": "PlantGuard is running"}

from pydantic import BaseModel
from typing import Optional

class PlantResponse(BaseModel):
    plant_status: str
    is_poisonous: Optional[bool]
    confidence: float
    toxic_probability: float
    nontoxic_probability: float
    advice: str

@app.post("/identify")
async def identify_plant(file: UploadFile = File(...)):
    # 1. Read and validate image
    try:
        image_bytes = await file.read()
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(400, "Image too large (max 10 MB).")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # optional: add size/ratio checks here if you want
    except Exception as e:
        raise HTTPException(400, f"Invalid image file: {str(e)}")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]

    prob_nontoxic = probs[1].item()
    prob_toxic = probs[0].item()
    confidence = max(prob_nontoxic, prob_toxic)

    if confidence < LOW_CONF_THRESHOLD:
        plant_status = "Unknown"
        is_poisonous = None
    else:
        predicted_idx = probs.argmax().item()
        plant_status = CLASS_NAMES[predicted_idx]
        is_poisonous = (plant_status == "Toxic")

    advice = CLASS_INFO[plant_status]["advice"]

    return {
        "plant_status": plant_status,
        "is_poisonous": is_poisonous,
        "confidence_percent": f"{confidence * 100:.1f}%",
        "toxic_percent": f"{prob_toxic * 100:.1f}%",
        "Non-toxic_percent": f"{prob_nontoxic * 100:.1f}%",
        "advice": advice,
    }
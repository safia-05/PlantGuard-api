from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PlantGuard",
    description="Upload a plant image and get a Toxic / Non-Toxic prediction.",
    version="1.0.0",
)

# Allow Flutter (or any client) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants (extracted from the training notebook) ───────────────────────────
IMG_SIZE = 260          # size used during training
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

# ── Model path ─────────────────────────────────────────────────────────────────
# Put whichever .pth file you want to use here.
# Use "best_efficientnet_b2.pth"  for the best checkpoint
# Use "efficientnet_b2_final.pth" for the final fine-tuned version
MODEL_PATH = "best_efficientnet_b2.pth"

# ── Build the exact same architecture the AI team used ─────────────────────────
def build_model():
    model = models.efficientnet_b2(weights=None)
    num_features = model.classifier[1].in_features  # 1408 for EfficientNet-B2
    
    # Match the architecture used during training
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),          # dropout rate may vary; 0.4 is common
        nn.Linear(num_features, 2)  # direct to 2 classes
    )
    return model

# ── Load model once at startup ─────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = build_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print(f"Model loaded from '{MODEL_PATH}' on {device}")

# ── Preprocessing (eval transform from the notebook) ──────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    ),
])

# ── Routes ─────────────────────────────────────────────────────────────────────
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
    """
    Upload a plant image (JPG / PNG) and receive a prediction.

    Returns:
        - plant_status : "Non-Toxic" | "Toxic" | "Unknown"
        - is_poisonous : true / false / null (null = unknown)
        - confidence   : 0.0 – 1.0
        - toxic_probability   : probability the plant is toxic
        - nontoxic_probability: probability the plant is non-toxic
        - advice       : short safety advice string
    """
    # ── Validate file type ────────────────────────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported.",
        )

    # ── Read & preprocess image ───────────────────────────────────────────────
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the image file.")

    tensor = transform(image).unsqueeze(0).to(device)  # shape: [1, 3, 260, 260]

    # ── Run inference ─────────────────────────────────────────────────────────
    with torch.no_grad():
        outputs = model(tensor)                        # raw logits
        probs   = F.softmax(outputs, dim=1)[0]         # probabilities for class 0 & 1

    prob_nontoxic = probs[0].item()
    prob_toxic    = probs[1].item()
    confidence    = max(prob_nontoxic, prob_toxic)

    # ── Apply confidence thresholds (mirrors the notebook logic) ─────────────
    if confidence < LOW_CONF_THRESHOLD:
        plant_status = "Unknown"
        is_poisonous = None
    else:
        predicted_idx = probs.argmax().item()
        plant_status  = CLASS_NAMES[predicted_idx]
        is_poisonous  = (plant_status == "Toxic")

    advice = CLASS_INFO[plant_status]["advice"]

    return {
        "plant_status"         : plant_status,
        "is_poisonous"         : is_poisonous,
        "confidence"           : round(confidence, 4),
        "toxic_probability"    : round(prob_toxic, 4),
        "nontoxic_probability" : round(prob_nontoxic, 4),
        "advice"               : advice,
    }
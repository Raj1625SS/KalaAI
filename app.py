from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import io

# -----------------------------
# Load checkpoint
# -----------------------------
MODEL_PATH = "models/deit_textile_best.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ckpt = torch.load(MODEL_PATH, map_location=device)

CLASSES = ckpt["class_names"]

model = timm.create_model(
    "deit_small_patch16_224",
    pretrained=False,
    num_classes=len(CLASSES)
)

model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(
    title="KalaAI Textile Classifier",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message": "KalaAI API running",
        "classes": CLASSES
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]

    class_probs = {
        CLASSES[i]: round(float(probs[i]), 4)
        for i in range(len(CLASSES))
    }

    pred_idx = probs.argmax().item()

    return {
        "pattern": CLASSES[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "class_probabilities": class_probs
    }
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import io
 
# -----------------------------
# Cultural story data
# -----------------------------
PATTERN_INFO = {
    "gamthi": {
        "origin": "Gujarat, India — practiced by Khatri artisan communities",
        "region": "Kutch and Saurashtra, Gujarat",
        "motifs": "Bold geometric shapes, mirror-work inspired patterns, tribal folk symbols",
        "significance": "Each block is hand-carved from teak wood by master craftsmen. A single saree can take 2–3 weeks to complete by hand.",
        "at_risk": True
    },
    "kalamkari": {
        "origin": "Over 3000 years old — mentioned in ancient Sanskrit texts",
        "region": "Srikalahasti and Machilipatnam, Andhra Pradesh",
        "motifs": "Scenes from Ramayana and Mahabharata, lotus flowers, peacocks",
        "significance": "One of India's oldest textile traditions. Natural dyes from pomegranate, indigo, and iron are used. GI tagged by the Indian government.",
        "at_risk": False
    },
    "mughal": {
        "origin": "Introduced during the Mughal Empire (16th–19th century)",
        "region": "Agra, Delhi, Lucknow — North India",
        "motifs": "Persian-inspired florals, arabesque patterns, symmetrical garden designs",
        "significance": "Patronized by Mughal emperors including Akbar and Jahangir. Blends Indian and Persian artistic traditions into a unique hybrid style.",
        "at_risk": False
    },
    "sanganeri": {
        "origin": "Developed in Sanganer town, 16th century",
        "region": "Sanganer, near Jaipur, Rajasthan",
        "motifs": "Delicate floral sprigs, butis, fine vines on white or pastel backgrounds",
        "significance": "Sanganer has over 100 block printing families. The River Saraswati historically provided mineral-rich water essential for the dyeing process.",
        "at_risk": True
    }
}
 
# -----------------------------
# Load checkpoint
# -----------------------------
MODEL_PATH = "models/deit_textile_best.pth"
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
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
    predicted_class = CLASSES[pred_idx]
 
    story = PATTERN_INFO.get(predicted_class.lower(), {
        "origin": "Traditional Indian textile",
        "region": "India",
        "motifs": "Traditional patterns",
        "significance": "Part of India's rich textile heritage.",
        "at_risk": False
    })
 
    return {
        "pattern": predicted_class,
        "confidence": round(float(probs[pred_idx]), 4),
        "class_probabilities": class_probs,
        "story": story
    }
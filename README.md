# 🪡 KalaAI — Textile Heritage Intelligence

> **Preserving India's block print traditions through Explainable AI**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev)
[![Model](https://img.shields.io/badge/Model-DeiT_ViT-orange.svg)](https://huggingface.co/facebook/deit-small-patch16-224)
[![Accuracy](https://img.shields.io/badge/Accuracy-92.9%25-brightgreen.svg)]()

---

## 📌 The Problem

India has over **3,000 years of handmade textile heritage** — Mughal, Sanganeri, Kalamkari, Gamthi, and dozens more traditions, each telling a unique story of region, community, and craft.

Yet today:
- Most people cannot distinguish one block print tradition from another
- Artisan communities are shrinking and knowledge is not being passed down
- Machine-printed copies flood markets, making authentic handmade textiles impossible to verify
- No scalable AI system exists for identification, preservation, or digital archiving of these traditions

> *"What cannot be identified cannot be preserved."*

---

## 💡 What is KalaAI?

**KalaAI** is an AI-powered cultural heritage platform that:

- 🔍 **Identifies** the block print tradition from any textile image
- 📖 **Explains** the cultural origin, motifs, and historical significance
- ⚠️ **Flags** traditions that are endangered or at risk of disappearing
- 🌐 **Exposes** a live API ready for integration with museums, marketplaces, and tourism platforms

Built for **Tradition Hacks 2026** — under the theme of *AI for Cultural Preservation*.

---

## 🎯 Supported Textile Traditions

| Tradition | Region | Status |
|---|---|---|
| **Mughal** | Agra, Delhi, Lucknow — North India | Stable |
| **Sanganeri** | Sanganer, Jaipur, Rajasthan | ⚠️ At Risk |
| **Kalamkari** | Andhra Pradesh & Telangana | Stable |
| **Gamthi** | Kutch & Saurashtra, Gujarat | ⚠️ At Risk |

---

## ✨ Features

- **Vision Transformer Classification** — Fine-tuned DeiT (Data-efficient Image Transformer) achieving 92.9% validation accuracy
- **Cultural Story Engine** — Each prediction surfaces origin, region, motifs, and heritage significance
- **Heritage Risk Flagging** — Highlights traditions that are endangered
- **Confidence Visualization** — Probability bars for all classes
- **Live REST API** — FastAPI backend deployed on Hugging Face Spaces
- **Responsive Web UI** — React frontend with drag-and-drop image upload

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| AI Model | DeiT (Data-efficient Vision Transformer) via `timm` |
| Training | PyTorch, torchvision |
| Backend | FastAPI, Uvicorn |
| Frontend | React, Vite, Axios |
| Deployment | Hugging Face Spaces (backend), Vercel (frontend) |
| Data | Custom dataset — 4 Indian textile traditions |

---

## 📁 Project Structure

```
KalaAI/
├── app.py                  # FastAPI backend
├── requirements.txt        # Local dependencies (CUDA)
├── requirements_deploy.txt # Production dependencies (CPU)
├── Dockerfile              # HuggingFace deployment config
├── collect_data.py         # Dataset collection script
├── prepare_data.py         # Train/val/test split script
├── augment_data.py         # Data augmentation script
├── models/
│   └── deit_textile_best.pth  # Trained model checkpoint
├── data/
│   ├── raw/                # Original collected images
│   ├── train/              # Training set (75%)
│   ├── val/                # Validation set (15%)
│   └── test/               # Test set (10%)
└── frontend/
    ├── src/
    │   ├── App.jsx         # Main React component
    │   └── App.css         # Styles
    └── package.json
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### 1. Clone the repository
```bash
git clone https://github.com/Raj7754SS/KalaAI.git
cd KalaAI
```

### 2. Set up Python environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Start the backend
```bash
python -m uvicorn app:app --reload
```
Backend runs at: `http://127.0.0.1:8000`
Interactive API docs: `http://127.0.0.1:8000/docs`

### 4. Start the frontend
```bash
cd frontend
npm install
npm run dev
```
Frontend runs at: `http://localhost:5173`

---

## 🌐 Live Demo

| Service | URL |
|---|---|
| Frontend | `https://kalaai.vercel.app` |
| Backend API | `https://sayan77-kalaai-backend.hf.space` |
| API Docs | `https://sayan77-kalaai-backend.hf.space/docs` |

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Validation Accuracy | **92.9%** |
| Model Architecture | DeiT-Small (Patch 16, 224x224) |
| Training Images | ~800 (after augmentation: 3,200+) |
| Classes | 4 Indian textile traditions |
| Training Device | CUDA GPU |

---

## 🔌 API Reference

### `GET /`
Returns API status and available classes.

**Response:**
```json
{
  "message": "KalaAI API running",
  "classes": ["gamthi", "kalamkari", "mughal", "sanganeri"]
}
```

### `POST /predict`
Classifies a textile image and returns cultural information.

**Request:** `multipart/form-data` with `file` field (JPG, PNG, WEBP)

**Response:**
```json
{
  "pattern": "Mughal",
  "confidence": 0.98,
  "class_probabilities": {
    "gamthi": 0.001,
    "kalamkari": 0.012,
    "mughal": 0.98,
    "sanganeri": 0.007
  },
  "story": {
    "origin": "Introduced during the Mughal Empire (16th–19th century)",
    "region": "Agra, Delhi, Lucknow — North India",
    "motifs": "Persian-inspired florals, arabesque patterns, symmetrical garden designs",
    "significance": "Patronized by Mughal emperors including Akbar and Jahangir.",
    "at_risk": false
  }
}
```

---

## 🗺️ Roadmap

- [x] Dataset collection and cleaning
- [x] DeiT model training (92.9% accuracy)
- [x] FastAPI backend with cultural story engine
- [x] React frontend with confidence visualization
- [x] Deployment on Hugging Face + Vercel
- [ ] Authenticity detector (handmade vs machine-printed)
- [ ] Blockchain certificate for artisan verification
- [ ] Expand to 20+ Indian textile traditions
- [ ] Mobile app for tourist use

---

## 🙏 Acknowledgements

- [Facebook Research](https://github.com/facebookresearch/deit) for the DeiT model architecture
- [Hugging Face](https://huggingface.co) for model hosting
- Indian artisan communities whose craft inspired this project
- Tradition Hacks 2026 for the platform to build this

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Sayan** — [GitHub](https://github.com/Raj7754SS) · [LinkedIn](www.linkedin.com/in/sayan-sahoo-0141b9300)

---

*Built with ❤️ for India's textile artisans — Tradition Hacks 2026*

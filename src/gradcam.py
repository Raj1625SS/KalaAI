import torch
import numpy as np
import cv2
import timm
from torchvision import transforms
from PIL import Image
import os

# ---------------- CONFIG ----------------

CHECKPOINT = "../models/deit_textile_best.pth"
OUTPUT_DIR = "../outputs/gradcam"
NUM_CLASSES = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------

ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
class_names = ckpt["class_names"]

model = timm.create_model(
    "deit_small_patch16_224",
    pretrained=False,
    num_classes=NUM_CLASSES
)

model.load_state_dict(ckpt["model_state"])
model = model.to(DEVICE)
model.eval()

# ---------------- TRANSFORM ----------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- FEATURE MAP EXTRACTION ----------------

feature_maps = []

def feature_hook(module, input, output):
    feature_maps.append(output.detach())

# Patch embedding output
hook = model.patch_embed.register_forward_hook(feature_hook)

# ---------------- HEATMAP ----------------

def generate_heatmap(image_path, save_name=None):

    feature_maps.clear()

    pil_img = Image.open(image_path).convert("RGB")
    pil_img = pil_img.resize((224, 224))

    rgb_img = np.array(pil_img)

    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()
    pred_class = class_names[pred_idx]

    # Feature map shape:
    # [1, 384, 14, 14]

    fmap = feature_maps[0][0]

    if len(fmap.shape) == 3:
        fmap = fmap.mean(dim=0).cpu().numpy()
    else:
        fmap = fmap.mean(dim=0).cpu().numpy()

    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)

    heatmap = cv2.resize(
        fmap,
        (224, 224),
        interpolation=cv2.INTER_CUBIC
    )

    heatmap_uint8 = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        rgb_img,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    combined = np.hstack([rgb_img, overlay])

    output_name = save_name or os.path.splitext(
        os.path.basename(image_path)
    )[0]

    out_path = os.path.join(
        OUTPUT_DIR,
        f"{output_name}__{pred_class}.jpg"
    )

    cv2.imwrite(
        out_path,
        cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    )

    print(
        f"{os.path.basename(image_path):30s} "
        f"-> {pred_class:12s} "
        f"conf={confidence:.3f}"
    )

    return out_path

# ---------------- MAIN ----------------

if __name__ == "__main__":

    test_dir = "../data/test"

    print("\nGenerating heatmaps...\n")

    for class_name in sorted(os.listdir(test_dir)):

        class_path = os.path.join(test_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        print(f"\n--- {class_name} ---")

        for img_file in sorted(os.listdir(class_path)):

            if not img_file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            img_path = os.path.join(
                class_path,
                img_file
            )

            save_name = (
                f"{class_name}_"
                f"{os.path.splitext(img_file)[0]}"
            )

            generate_heatmap(
                img_path,
                save_name
            )

    print("\nDone!")
    print(f"Saved in: {OUTPUT_DIR}")
import torch

ckpt = torch.load(
    "../models/deit_textile_best.pth",
    map_location="cpu"
)

print("Checkpoint loaded successfully!")
print("Keys:", ckpt.keys())
print("Classes:", ckpt["class_names"])
print("Validation Accuracy:", ckpt["val_acc"])
import os, torch, torch.nn as nn
import timm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
from dataset import get_dataloaders

# ─── Config ───────────────────────────────────────────────
DATA_DIR    = "../data"
SAVE_PATH   = "../models/deit_textile_best.pth"
NUM_CLASSES = 4
EPOCHS      = 30
BATCH_SIZE  = 16
LR          = 2e-4
# ──────────────────────────────────────────────────────────

def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss_sum += criterion(logits, labels).item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)
            preds_all.extend(pred.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
    return loss_sum / len(loader), correct / total, preds_all, labels_all


if __name__ == '__main__':          # ← required on Windows
    os.makedirs("../models", exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}  ({torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU'})")

    train_loader, val_loader, test_loader, class_names = \
        get_dataloaders(DATA_DIR, BATCH_SIZE)

    # timm loads DeiT cleanly — no key mismatch warnings
    model = timm.create_model(
        "deit_small_patch16_224",
        pretrained=True,
        num_classes=NUM_CLASSES
    )
    model = model.to(DEVICE)

    # Phase 1: freeze backbone, train head only
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=LR, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val_acc = 0.0

    for epoch in range(EPOCHS):

        # Phase 2: unfreeze everything at epoch 5
        if epoch == 5:
            print("\n🔓 Unfreezing full model...\n")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = AdamW(model.parameters(), lr=LR * 0.1, weight_decay=0.01)

        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += labels.size(0)

        train_acc = correct / total
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        marker = "  ✓ saved" if val_acc > best_val_acc else ""
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train {train_acc:.3f} | Val {val_acc:.3f} | "
              f"VLoss {val_loss:.4f}{marker}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     val_acc,
                "class_names": class_names
            }, SAVE_PATH)

    # Final test evaluation
    print(f"\n{'═'*50}")
    ckpt = torch.load(SAVE_PATH)
    model.load_state_dict(ckpt["model_state"])
    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, DEVICE)
    print(f"  Best val acc : {ckpt['val_acc']:.4f}  (epoch {ckpt['epoch']+1})")
    print(f"  Test acc     : {test_acc:.4f}")
    print(f"\n{classification_report(test_labels, test_preds, target_names=class_names)}")
    print("Confusion matrix:")
    print(confusion_matrix(test_labels, test_preds))
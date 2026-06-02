import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, shear=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_dataloaders(data_dir: str, batch_size: int = 16):
    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_transforms)
    val_ds   = datasets.ImageFolder(f"{data_dir}/val",   transform=val_transforms)
    test_ds  = datasets.ImageFolder(f"{data_dir}/test",  transform=val_transforms)

    class_counts   = np.bincount(train_ds.targets)
    weights        = 1.0 / class_counts
    sample_weights = [weights[t] for t in train_ds.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=0)   # ← 0 for Windows
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=0)

    print(f"\n{'─'*45}")
    print(f"  Classes : {train_ds.classes}")
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    print(f"  Class counts: {dict(zip(train_ds.classes, class_counts))}")
    print(f"{'─'*45}\n")

    return train_loader, val_loader, test_loader, train_ds.classes
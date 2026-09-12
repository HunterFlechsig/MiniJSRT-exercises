import math
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from utils.utils import get_data_dir, get_output_dir

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)

DATA_DIR = get_data_dir() / "Age_RGB"

TRAIN_CSV = DATA_DIR / "trainingdata.csv"
TEST_CSV = DATA_DIR / "testdata.csv"
IMAGE_DIR = DATA_DIR / "JPGs"

OUTPUT_DIR = get_output_dir() / "age_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

class XRayAgeDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform

        cols = {c.lower(): c for c in self.df.columns}
        self.filename_col = cols["filenames"]
        self.age_col = cols["age"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(
            self.image_dir / row[self.filename_col]
        ).convert("RGB")

        if self.transform:
            img = self.transform(img)

        age = torch.tensor(
            float(row[self.age_col]),
            dtype=torch.float32
        )

        return img, age

train_dataset = XRayAgeDataset(
    TRAIN_CSV,
    IMAGE_DIR,
    train_transform
)

test_dataset = XRayAgeDataset(
    TEST_CSV,
    IMAGE_DIR,
    test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(train_dataset.df.head())
print(train_dataset.df["age"].describe())

model = models.resnet18(
    weights=models.ResNet18_Weights.IMAGENET1K_V1
)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
)

model = model.to(device)

criterion = nn.SmoothL1Loss()

optimizer = torch.optim.AdamW(
    model.fc.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)

best_mae = float("inf")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    if epoch == 10: 
        for param in model.parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-5,
            weight_decay=1e-4,
        )

    for images, ages in train_loader:

        images = images.to(device)
        ages = ages.to(device)

        optimizer.zero_grad()

        preds = model(images).squeeze(1)

        loss = criterion(preds, ages)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_dataset)

    model.eval()

    abs_error = 0.0
    sq_error = 0.0

    with torch.no_grad():

        for images, ages in test_loader:

            images = images.to(device)
            ages = ages.to(device)

            preds = model(images).squeeze(1)

            abs_error += torch.abs(preds - ages).sum().item()
            sq_error += ((preds - ages) ** 2).sum().item()

    mae = abs_error / len(test_dataset)
    rmse = math.sqrt(sq_error / len(test_dataset))

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} "
        f"TrainLoss={train_loss:.4f} "
        f"MAE={mae:.2f} "
        f"RMSE={rmse:.2f}"
    )

    if mae < best_mae:
        best_mae = mae
        torch.save(
            model.state_dict(),
            OUTPUT_DIR / "best_resnet18_age.pth"
        )

print("\nTraining complete.")
print("Best MAE:", round(best_mae,2))

model.load_state_dict(
    torch.load(
        OUTPUT_DIR / "best_resnet18_age.pth",
        map_location=device
    )
)

model.eval()

print("\nSample predictions\n")

for i in range(min(10, len(test_dataset))):

    image, age = test_dataset[i]

    with torch.no_grad():
        pred = model(
            image.unsqueeze(0).to(device)
        ).item()

    print(
        f"{i+1:02d}: "
        f"True={age.item():.0f}  "
        f"Predicted={pred:.1f}"
    )

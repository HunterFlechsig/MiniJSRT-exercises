import torch
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path

from utils.utils import get_data_dir
from utils.training_utils import better_train_model, evaluate_model

# -------------------------
# Device
# -------------------------
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# -------------------------
# Directories
# -------------------------
data_dir = get_data_dir() / "Gender01_RGB"
train_dir = data_dir / "train"
test_dir = data_dir / "test"

output_dir = Path.cwd() / "gender"
output_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# Hyperparameters
# -------------------------
BATCH_SIZE = 16
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
NUM_CLASSES = 2

# -------------------------
# Data transforms
# -------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),

    # Uncomment if your images are grayscale
    # transforms.Grayscale(num_output_channels=3),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # Uncomment if grayscale
    # transforms.Grayscale(num_output_channels=3),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# -------------------------
# Datasets
# -------------------------
train_dataset = ImageFolder(train_dir, transform=train_transform)
test_dataset = ImageFolder(test_dir, transform=test_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# -------------------------
# Model
# -------------------------
weights = models.ResNet18_Weights.IMAGENET1K_V1

model = models.resnet18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

# -------------------------
# Optimizer / Loss
# -------------------------
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)

criterion = torch.nn.CrossEntropyLoss()

# -------------------------
# Train
# -------------------------
better_train_model(
    model=model,
    train_loader=train_loader,
    num_epochs=NUM_EPOCHS,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
)

# -------------------------
# Save
# -------------------------
torch.save(
    model.state_dict(),
    output_dir / "gender_model_weights.pth",
)

print(f"Model saved to: {output_dir / 'gender_model_weights.pth'}")

# -------------------------
# Evaluate
# -------------------------
evaluate_model(
    model=model,
    device=device,
    test_loader=test_loader,
)
import torch
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path
from utils.utils import ( get_data_dir, get_current_dir )
from utils.training_utils import ( train_model, evaluate_model )

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

data_dir = get_data_dir() / "direction" / "Directions01_RGB"
train_dir = data_dir / "train"
test_dir = data_dir / "test"
output_dir = Path.cwd() / "direction"

weights = models.ResNet18_Weights.DEFAULT
num_classes = 4

# load data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = ImageFolder(
    root=train_dir,
    transform=transform
)

test_dataset = ImageFolder(
    root=test_dir,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# create ResNet18
model = models.resnet18(weights = weights)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()
num_epochs = 5

train_model(
    model=model,
    train_loader=train_loader,
    num_epochs=num_epochs,
    criterion=criterion,
    optimizer=optimizer,
    device=device
)

# save weights for later analysis
torch.save(
    model.state_dict(),
    output_dir / "direction_model_weights.pth"
)

evaluate_model(
    model=model,
    device=device,
    test_loader=test_loader
)



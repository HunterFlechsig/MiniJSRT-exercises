import torch
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from utils.gradcam_utils import generate_gradcam
from utils.utils import get_data_dir, get_output_dir
from utils.training_utils import better_train_model, evaluate_model

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

data_dir = get_data_dir() / "Gender01_RGB"
train_dir = data_dir / "train"
test_dir = data_dir / "test"

output_dir = get_output_dir() / "gender"
output_dir.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_CLASSES = 2

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

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

weights = models.ResNet18_Weights.IMAGENET1K_V1

model = models.resnet18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)

criterion = torch.nn.CrossEntropyLoss()

better_train_model(
    model=model,
    train_loader=train_loader,
    num_epochs=NUM_EPOCHS,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
)

torch.save(
    model.state_dict(),
    output_dir / "gender_model_weights.pth",
)

print(f"Model saved to: {output_dir / 'gender_model_weights.pth'}")

evaluate_model(
    model=model,
    device=device,
    test_loader=test_loader,
)

model.eval()

class_names = test_dataset.classes

for i in range(5):

    image, label = test_dataset[i]

    input_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(input_tensor)

        prediction = output.argmax(dim=1).item()

    print(
        f"Image {i}: "
        f"True={class_names[label]}, "
        f"Predicted={class_names[prediction]}"
    )

    generate_gradcam(
        model=model,
        image_tensor=input_tensor,
        predicted_class=prediction,
        save_path=output_dir / f"gradcam_{i}.png",
    )

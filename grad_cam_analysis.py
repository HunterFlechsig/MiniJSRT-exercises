import torch
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.gradcam_utils import generate_gradcam
from utils.utils import get_data_dir, get_output_dir

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

output_dir = get_output_dir() / "gender"
output_dir.mkdir(parents=True, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

test_dataset = ImageFolder(
    get_data_dir() / "Gender01_RGB" / "test",
    transform=transform
)

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

model.load_state_dict(
    torch.load(
        output_dir / "gender_model_weights.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

classes = test_dataset.classes

wrong = 0

for i, (image, label) in enumerate(test_dataset):

    input_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.argmax(dim=1).item()

    if prediction != label:

        print(
            f"Wrong prediction {wrong+1}\n"
            f"Image: {i}\n"
            f"True: {classes[label]}\n"
            f"Predicted: {classes[prediction]}"
        )

        generate_gradcam(
            model,
            input_tensor,
            prediction,
            output_dir / f"wrong_{wrong}.png"
        )

        wrong += 1

    if wrong == 5:
        break
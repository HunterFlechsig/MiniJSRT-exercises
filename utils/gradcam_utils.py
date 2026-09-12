import numpy as np
import torch
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(image):
    image = image.cpu().numpy().transpose(1, 2, 0)

    image = image * IMAGENET_STD + IMAGENET_MEAN

    image = np.clip(image, 0, 1)

    return image


def generate_gradcam(
    model,
    image_tensor,
    predicted_class,
    save_path,
):
    model.eval()

    target_layers = [model.layer4[-1]]

    with GradCAM(
        model=model,
        target_layers=target_layers,
    ) as cam:

        targets = [ClassifierOutputTarget(predicted_class)]

        grayscale_cam = cam(
            input_tensor=image_tensor,
            targets=targets,
        )[0]

    original = denormalize(image_tensor.squeeze())

    visualization = show_cam_on_image(
        original,
        grayscale_cam,
        use_rgb=True,
    )

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(original)
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(grayscale_cam, cmap="jet")
    ax[1].set_title("Grad-CAM")
    ax[1].axis("off")

    ax[2].imshow(visualization)
    ax[2].set_title("Overlay")
    ax[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
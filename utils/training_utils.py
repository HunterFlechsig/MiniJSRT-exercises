import torch
from sklearn.metrics import accuracy_score
import copy

def train_model(model, train_loader, num_epochs, criterion, optimizer, device):
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.float() / len(train_loader.dataset)

        print(f'Epoch [{epoch+1}/{num_epochs}], train loss: {train_loss:.4f}, train acc: {train_acc:.4f}')

def better_train_model(
    model,
    train_loader,
    num_epochs,
    criterion,
    optimizer,
    device,
    scheduler=None,
):
    model = model.to(device)

    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, dim=1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:2d}/{num_epochs} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Accuracy: {epoch_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

    return train_losses, train_accuracies

def evaluate_model(model, test_loader, device):
    model = model.to(device)
    correct_pred = {classname: 0 for classname in test_loader.dataset.classes}
    total_pred = {classname: 0 for classname in test_loader.dataset.classes}

    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            for label, prediction in zip(labels, preds):
                classname = test_loader.dataset.classes[label]
                if label == prediction:
                    correct_pred[classname] += 1
                total_pred[classname] += 1

    accuracy_per_class = {classname: correct_pred[classname] / total_pred[classname] if total_pred[classname] > 0 else 0
                          for classname in test_loader.dataset.classes}

    overall_accuracy = accuracy_score(all_labels, all_preds)

    print("Accuracy per class:")
    for classname, accuracy in accuracy_per_class.items():
        print(f"{classname}: {accuracy:.4f}")

    print()
    print(f"Overall Accuracy: {overall_accuracy:.4f}")

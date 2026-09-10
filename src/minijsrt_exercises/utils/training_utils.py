import torch
from sklearn.metrics import accuracy_score

def train_model(model, train_loader, num_epochs, criterion, optimizer, device):
    model = model.to(device)

    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()

        # Initialize running loss and correct predictions count for training
        running_loss = 0.0
        running_corrects = 0

        # Iterate over the training data loader
        for inputs, labels in train_loader:
            # Move inputs and labels to the device (GPU or CPU)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Reset the gradients to zero before the backward pass
            optimizer.zero_grad()

            # Forward pass: compute the model output
            outputs = model(inputs)
            # Get the predicted class (with the highest score)
            _, preds = torch.max(outputs, 1)
            # Compute the loss between the predictions and actual labels
            loss = criterion(outputs, labels)

            # Backward pass: compute gradients
            loss.backward()
            # Perform the optimization step to update model parameters
            optimizer.step()

            # Accumulate the running loss and the number of correct predictions
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # Compute average training loss and accuracy for this epoch
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.float() / len(train_loader.dataset)

        print(f'Epoch [{epoch+1}/{num_epochs}], train loss: {train_loss:.4f}, train acc: {train_acc:.4f}')

def evaluate_model(model, test_loader, device):
    model = model.to(device)
    # Initialize dictionaries to store correct and total predictions
    correct_pred = {classname: 0 for classname in test_loader.dataset.classes}
    total_pred = {classname: 0 for classname in test_loader.dataset.classes}

    # Set the model to evaluation mode
    model.eval()

    # Track the ground truth labels and predictions
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Collect predictions and labels for metric calculations
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            # Update the correct and total predictions
            for label, prediction in zip(labels, preds):
                classname = test_loader.dataset.classes[label]
                if label == prediction:
                    correct_pred[classname] += 1
                total_pred[classname] += 1

    # Calculate accuracy per class
    accuracy_per_class = {classname: correct_pred[classname] / total_pred[classname] if total_pred[classname] > 0 else 0
                          for classname in test_loader.dataset.classes}

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(all_labels, all_preds)


    # Print the evaluation results
    print("Accuracy per class:")
    for classname, accuracy in accuracy_per_class.items():
        print(f"{classname}: {accuracy:.4f}")

    print()
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
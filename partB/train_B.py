
import os
import shutil
import random
from collections import defaultdict
import wandb
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.models as models
import argparse

wandb.login()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_loaders(image_size,train_data_dir, test_data_dir, data_augmentation):
    # Define default transform
    default_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),

    ])

    # Define transform for data augmentation

    augment_transform = transforms.Compose([
        transforms.RandomRotation(degrees=(5,5), fill=(0,)),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size)

    ])


    # Load the dataset without augmentation
    train_dataset = datasets.ImageFolder(root=train_data_dir, transform=default_transform)

    # Separate classes
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

    # Create a dictionary to hold data indices for each class
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(train_dataset.samples):
        class_indices[label].append(idx)

    # Set aside 20% of data from each class for validation
    validation_indices = []
    train_indices = []
    for class_idx, indices in class_indices.items():
        num_validation = int(0.2 * len(indices))
        random.shuffle(indices)  # Shuffle indices to ensure randomness
        validation_indices.extend(indices[:num_validation])
        train_indices.extend(indices[num_validation:])

    # Create PyTorch data loaders for the initial dataset
    train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=32, shuffle=True)
    validation_loader = DataLoader(Subset(train_dataset, validation_indices), batch_size=32, shuffle=True)

    # Load test data
    test_dataset = datasets.ImageFolder(root=test_data_dir, transform=default_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # If data augmentation is enabled, create augmented data loader
    if data_augmentation:
        # Create DataLoader for augmented training data using train_indices
        augmented_dataset = datasets.ImageFolder(root=train_data_dir, transform=augment_transform)
        augmented_loader = DataLoader(Subset(augmented_dataset, train_indices), batch_size=32, shuffle=True)
        # Combine original training data and augmented data

        combined_dataset = ConcatDataset([train_loader.dataset, augmented_loader.dataset])
        train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

    return train_loader, validation_loader, test_loader

     

def CNN_ResNet50(num_classes, batch_norm, dropout_rate, activation, dense_neurons, strategy):
    # Define activation function
    act_fn = {
        'ReLU': nn.ReLU(),
        'GELU': nn.GELU(),
        'SiLU': nn.SiLU(),
        'Mish': nn.Mish()
    }[activation]

    # Load pre-trained ResNet50 model
    resnet = models.resnet50(pretrained=True)

    # Determine the layers to freeze based on the selected strategy
    if strategy == 'feature_extraction':
        # Freeze all parameters except the final classifier
        for param in resnet.parameters():
            param.requires_grad = False
    elif strategy == 'fine_tuning':
        # Fine-tune the entire model No need to freeze any layers
        pass
    elif strategy == 'fine_tuning_partial':
        # Fine-tune up to a certain layer (including the specified layer and all layers above it)
        # Here, I fine-tune from layer5 block (5th layer)
        unfreeze_from_layer = 5
        for idx, (name, param) in enumerate(resnet.named_parameters()):
            if idx < unfreeze_from_layer:
                param.requires_grad = False
            else:
                param.requires_grad = True
    elif strategy == 'progressive_unfreezing':
        # Progressive unfreezing
        # Start by freezing all layers except the final classifier
        for param in resnet.parameters():
            param.requires_grad = False

        # Define the number of layers to unfreeze at each step
        num_layers_to_unfreeze = 4

        # Get the children of the resnet model
        resnet_children = list(resnet.children())

        # Loop through each layer and unfreeze progressively
        idx = 0
        while idx < len(resnet_children):
            unfreeze_layers = resnet_children[-(idx + num_layers_to_unfreeze):]
            for layer in unfreeze_layers:
                for param in layer.parameters():
                    param.requires_grad = True

            # Retain gradients of previously unfrozen layers
            if idx > 0:
                previous_unfreeze_layers = resnet_children[-(idx + num_layers_to_unfreeze + num_layers_to_unfreeze):-(idx + num_layers_to_unfreeze)]
                for layer in previous_unfreeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = True
            idx += num_layers_to_unfreeze

    else:
        raise ValueError("Invalid strategy. Please choose one of: 'feature_extraction', 'fine_tuning', 'fine_tuning_partial', 'progressive_unfreezing'")

    # Replace the last fully connected layer with a new one that has `dense_neurons` output features
    num_ftrs = resnet.fc.in_features
    layers = [
        nn.Linear(num_ftrs, dense_neurons),
        act_fn
    ]
    if batch_norm:
        layers.append(nn.BatchNorm1d(dense_neurons))
    if dropout_rate > 0:
        layers.append(nn.Dropout(dropout_rate))
    layers.append(nn.Linear(dense_neurons, num_classes))
    resnet.fc = nn.Sequential(*layers)

    return resnet.to(device)

     

def train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs, device):
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        epoch_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for images, labels in epoch_progress:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            epoch_progress.set_postfix({'Loss': running_loss / len(train_loader), 'Accuracy': 100 * correct_predictions / total_predictions})
        epoch_progress.close()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate validation loss and accuracy
        val_loss /= len(validation_loader)
        val_accuracy = 100 * val_correct / val_total
        accuracy = 100 * correct_predictions / total_predictions
        loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], accuracy: {accuracy:.2f}, loss: {loss:.2f}, val_accuracy: {val_accuracy:.2f}, val_loss: {val_loss:.2f}")

        # Log to Weights & Biases
        wandb.log({
            "Epoch": epoch + 1,
            "Accuracy": round(accuracy, 2),
            "Loss": round(loss, 2),
            "Validation Accuracy": round(val_accuracy, 2),
            "Validation Loss": round(val_loss, 2)
        })
    return model
     

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()

    accuracy = (correct / total) * 100
    print(f"Accuracy on test set: {accuracy:.2f}%")
     

# Define the wandb configuration
def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser for Sweep Configuration")

    # Model Parameters
    parser.add_argument('--activation', type=str, choices=['ReLU', 'GELU', 'SiLU', 'Mish'], required=True)
    parser.add_argument('--data_augmentation', type=str, choices=['Yes','No'], required=True)
    parser.add_argument('--batch_normalization', type=str, choices=['Yes', 'No'], required=True)
    parser.add_argument('--dropout', type=float, choices=[0.2, 0.3], required=True)
    parser.add_argument('--dense_neurons', type=int, choices=[128, 256, 512, 1024], required=True)
    parser.add_argument('--epoch', type=int, choices=[5,10], required=True)
    parser.add_argument('--learning_rate', type=float, choices=[0.001, 0.0001], required=True)
    parser.add_argument('--strategy', type=str, choices=['feature_extraction', 'fine_tuning', 'fine_tuning_partial', 'progressive_unfreezing'], required=True)

    args = parser.parse_args()
    return args

def train(args):
    # Initialize wandb
    wandb.init(project="deep_learn_assignment_2" ,entity="cs23m063")
    # Set your hyperparameters from wandb config
    config = args

    wandb.run.name = f'ResNet50_activation_{config.activation}_data_augmentation_{config.data_augmentation}_batch_normalization_{config.batch_normalization}_dropout_{config.dropout}_dense_neurons_{config.dense_neurons}_learning_rate_{config.learning_rate}_epoch_{config.epoch}_strategy_{config.strategy}'

    data_augmentation = config.data_augmentation == 'Yes'
    batch_normalization = config.batch_normalization == 'Yes'

    # Example usage:
    train_data_dir = r"inaturalist_12K\\train"
    test_data_dir = r"inaturalist_12K\\val"
    train_loader, validation_loader, test_loader = get_data_loaders(224, train_data_dir, test_data_dir, data_augmentation)

    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(validation_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")

    # Example usage:
    model = CNN_ResNet50(10, batch_normalization, config.dropout, config.activation, config.dense_neurons, config.strategy)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    model = train_model(model, train_loader, validation_loader, criterion, optimizer, config.epoch, device)

    # Evaluate on Test data
    evaluate_model(model, test_loader, device)

args = parse_args()
train(args)
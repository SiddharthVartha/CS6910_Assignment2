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
import argparse

wandb.login()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to display a single image
def imshow(img, title=None):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Function to visualize a single image from a DataLoader
def visualize_single_image(loader, title=None):
    data_iter = iter(loader)
    images, labels = next(data_iter)
    imshow(images[8], title=title)



def get_data_loaders(train_data_dir, test_data_dir, data_augmentation):
    # Define default transform
    default_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
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
        transforms.Resize(256),
        transforms.CenterCrop(256)

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

    #visualize_single_image(train_loader, title='Without Augmentation')

    # If data augmentation is enabled, create augmented data loader
    if data_augmentation:
        # Create DataLoader for augmented training data using train_indices
        augmented_dataset = datasets.ImageFolder(root=train_data_dir, transform=augment_transform)
        augmented_loader = DataLoader(Subset(augmented_dataset, train_indices), batch_size=32, shuffle=True)
        # Combine original training data and augmented data

        #visualize_single_image(augmented_loader, title='With Augmentation')
        combined_dataset = ConcatDataset([train_loader.dataset, augmented_loader.dataset])
        train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

    return train_loader, validation_loader, test_loader

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

def plot_sample_predictions(model, test_loader, device, num_samples=10):
    classes = {0: 'Amphibia', 1: 'Animalia', 2: 'Arachnida', 3: 'Aves', 4: 'Fungi',
                  5: 'Insecta', 6: 'Mammalia', 7: 'Mollusca', 8: 'Plantae', 9: 'Reptilia'}
    classes_count=[0,0,0,0,0,0,0,0,0,0]
    model.eval()
    count=0
    current_index=0
    with torch.no_grad():
        fig, axs = plt.subplots(10, 3, figsize=(15, 30))
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for image, label, pred in zip(images, labels, predicted):
                class_idx = label.item()
                if class_idx==current_index and count < 30 and classes_count[current_index]<3:
                    classes_count[current_index]+=1
                    image = image.cpu()  # Move image to CPU
                    axs[current_index, count%3].set_title(f"\nActual: {classes[current_index]}\nPredicted: {classes[pred.item()]} ")
                    axs[current_index, count%3].axis('off')
                    axs[current_index, count%3].imshow(image.permute(1, 2, 0).numpy())
                    count+=1

                if(classes_count[current_index]==3):
                        current_index+=1
                        break
            if count >= 30:
                break

    plt.tight_layout()
    image = wandb.Image(plt)
    wandb.log({"Predictions On Test": image})
    plt.show()

def CNN(input_channels, num_classes, num_filters, filter_size, activation, batch_norm, dropout, filter_multiplier, dense_neurons):
    # Define activation function
    act_fn = {
        'ReLU': nn.ReLU(),
        'GELU': nn.GELU(),
        'SiLU': nn.SiLU(),
        'Mish': nn.Mish()
    }[activation]

    layers = []
    in_channels = input_channels
    out_channels = num_filters
    for i in range(5):
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=filter_size[i], stride=1, padding=1),
            act_fn,
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_channels = out_channels
        out_channels = int(out_channels * filter_multiplier)

    # Calculate the size of the output after the convolutional layers
    x = torch.randn(1, input_channels, 256, 256)
    for layer in layers:
        x = layer(x)
    conv_out_size = x.size(1) * x.size(2) * x.size(3)

    model = nn.Sequential(
        *layers,
        nn.Flatten(),
        nn.Linear(conv_out_size, dense_neurons),
        act_fn,
        nn.Linear(dense_neurons, num_classes)
    )

    return model.to(device)

def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser for Sweep Configuration")

    # Model Parameters
    parser.add_argument('--num_filters', type=int, choices=[32, 64, 128], required=True)
    parser.add_argument('--activation', type=str, choices=['ReLU', 'GELU', 'SiLU', 'Mish'], required=True)
    parser.add_argument('--filter_size_1', type=int, choices=[2, 3, 5], required=True)
    parser.add_argument('--filter_size_2', type=int, choices=[2, 3, 5], required=True)
    parser.add_argument('--filter_size_3', type=int, choices=[2, 3, 5], required=True)
    parser.add_argument('--filter_size_4', type=int, choices=[2, 3, 5], required=True)
    parser.add_argument('--filter_size_5', type=int, choices=[2, 3, 5], required=True)
    parser.add_argument('--filter_multiplier', type=float, choices=[1, 2, 0.5], required=True)
    parser.add_argument('--data_augmentation', type=str, choices=['Yes', 'No'], required=True)
    parser.add_argument('--batch_normalization', type=str, choices=['Yes', 'No'], required=True)
    parser.add_argument('--dropout', type=float, choices=[0.2, 0.3], required=True)
    parser.add_argument('--dense_neurons', type=int, choices=[128, 256, 512, 1024], required=True)
    parser.add_argument('--epoch', type=int, choices=[5, 10], required=True)
    parser.add_argument('--learning_rate', type=float, choices=[0.001, 0.0001], required=True)

    args = parser.parse_args()
    return args


def train(args):
    # Initialize wandb
    wandb.init(project="deep_learn_assignment_2" ,entity="cs23m063")
    # Set your hyperparameters from wandb config
    config = args

    wandb.run.name = f'num_filters_{config.num_filters}_activation_{config.activation}_filter_multiplier_{config.filter_multiplier}_data_augmentation_{config.data_augmentation}_batch_normalization_{config.batch_normalization}_dropout_{config.dropout}_dense_neurons_{config.dense_neurons}_learning_rate_{config.learning_rate}_epoch_{config.epoch}'

    data_augmentation=False
    if(config.data_augmentation=='Yes'):
        data_augmentation=True

    batch_normalization=False
    if(config.batch_normalization=='Yes'):
        batch_normalization=True

    # Example usage:
    train_data_dir = r"inaturalist_12K\\train"
    test_data_dir = r"inaturalist_12K\\val"
    train_loader, validation_loader, test_loader = get_data_loaders(train_data_dir, test_data_dir, data_augmentation)

    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(validation_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    filter_size=[config.filter_size_1,config.filter_size_2,config.filter_size_3,config.filter_size_4,config.filter_size_5]
    # Example usage:
    model = CNN(
        input_channels=3,
        num_classes=10,
        num_filters=config.num_filters,
        filter_size=filter_size,
        activation=config.activation,
        batch_norm=batch_normalization,
        dropout=config.dropout,
        filter_multiplier=config.filter_multiplier,
        dense_neurons=config.dense_neurons
    )

    #model = CNN(input_channels=3, num_classes=10, num_filters=32, filter_size=3, activation='mish', batch_norm=True, dropout=0.2, filter_multiplier=1, dense_neurons=256)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    model=train_model(model, train_loader, validation_loader, criterion, optimizer, config.epoch, device)

    #Evaluate on Test data
    evaluate_model(model, test_loader, device)

    #Plot Test prediction
    #plot_sample_predictions(model, test_loader, device)

args = parse_args()
train(args)
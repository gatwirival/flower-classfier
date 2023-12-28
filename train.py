import argparse
import json
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a neural network on a given dataset.")
    parser.add_argument('data_directory', type=str, help='Path to the data directory for training.')
    parser.add_argument('--save_dir', type=str, help='Directory to save checkpoints.', default='checkpoints')
    parser.add_argument('--arch', type=str, help='Choose architecture.', default='vgg16')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for training.', default=0.01)
    parser.add_argument('--hidden_units', type=int, help='Number of hidden units in the classifier.', default=512)
    parser.add_argument('--epochs', type=int, help='Number of epochs for training.', default=20)
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training.')
    return parser.parse_args()

def load_data(data_directory):
    train_dir = os.path.join(data_directory, 'train')
    valid_dir = os.path.join(data_directory, 'valid')

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)

    return trainloader, validloader

def build_model(arch, hidden_units):
    model = getattr(models, arch)(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Define a new classifier
    if arch.startswith('vgg'):
        classifier_input_size = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
    elif arch.startswith('resnet'):
        classifier_input_size = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
    else:
        raise ValueError("Unsupported architecture. Please add handling for your specific architecture.")

    return model

def train_model(model, criterion, optimizer, trainloader, validloader, epochs, device):
    model.to(device)
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss, accuracy = validate_model(model, criterion, validloader, device)
                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss:.3f}.. "
                      f"Validation accuracy: {accuracy:.3f}")
                running_loss = 0
                model.train()

def validate_model(model, criterion, validloader, device):
    model.eval()
    valid_loss = 0
    accuracy = 0

    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model.forward(inputs)
            valid_loss += criterion(outputs, labels).item()

            ps = torch.exp(outputs)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

    model.train()
    return valid_loss/len(validloader), accuracy/len(validloader)

def save_checkpoint(model, optimizer, arch, hidden_units, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if arch.startswith('vgg'):
        checkpoint = {
            'arch': arch,
            'hidden_units': hidden_units,
            'classifier': model.classifier,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
    elif arch.startswith('resnet'):
        checkpoint = {
            'arch': arch,
            'hidden_units': hidden_units,
            'fc': model.fc,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
    else:
        raise ValueError("Unsupported architecture. Please add handling for your specific architecture.")

    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))

def main():
    args = parse_arguments()

    trainloader, validloader = load_data(args.data_directory)

    model = build_model(args.arch, args.hidden_units)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    train_model(model, criterion, optimizer, trainloader, validloader, args.epochs, device)

    save_checkpoint(model, optimizer, args.arch, args.hidden_units, args.save_dir)

if __name__ == '__main__':
    main()

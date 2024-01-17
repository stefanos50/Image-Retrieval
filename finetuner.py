import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label

def extract_class_number(file_path):
    # Use a regex pattern to match the number after "class_"
    pattern = re.compile(r'class_(\d+)')
    match = re.search(pattern, file_path)

    if match:
        # Extract the matched number
        class_number = int(match.group(1))
        return class_number
    else:
        # Return a default value or handle the case where the pattern is not found
        return None


# Define your data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

images_directory = "A:\\animals"  #https://www.kaggle.com/datasets/alessiocorrado99/animals10
class_labels = []
image_paths = []
class_names = ['dog','cat','elephant','butterfly','squirrel','sheep']
for root, dirs, files in os.walk(images_directory):
    for file in files:
        file_path = os.path.join(root, file)
        image_paths.append(file_path)
        class_labels.append(extract_class_number(file_path))
class_labels = np.array(class_labels)

train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, class_labels, test_size=0.2, random_state=42)

train_dataset = CustomDataset(train_paths, train_labels, transform=data_transforms['train'])
val_dataset = CustomDataset(val_paths, val_labels, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

model = models.resnet50(pretrained=True)
num_classes = len(set(train_labels))  # assuming train_labels are unique
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 5
best_val_accuracy = 0.0
best_model_weights = model.state_dict()

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            val_outputs = model(val_inputs)
            _, val_preds = torch.max(val_outputs, 1)

            all_val_preds.extend(val_preds.cpu().numpy())
            all_val_labels.extend(val_labels.cpu().numpy())

    val_accuracy = accuracy_score(all_val_labels, all_val_preds)
    print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}')

    # Save the model if it's the best so far
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_weights = model.state_dict()

# Save the best-performing model
torch.save(best_model_weights, 'best_finetuned.pth')

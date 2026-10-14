# EGR 425 - Object Recognition Project

# Dataset creation follows the PyTorch tutorial (https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
# As well as mirroring the pattern of the in-class example (ClassActivityHrsStudied_code.txt)
print("PROGRAM STARTED", flush=True)
from collections import Counter
from pathlib import Path
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import re
from torch import nn
import torch.nn.functional as F
from torch import optim


DATA_DIR = Path(__file__).resolve().parent / "ObjectImagesTrain"
BATCH_SIZE = 32
VAL_FRACTION = 0.20
SEED = 42 # makes the random split the same every time

IMAGE_EXTS = {".jpg"}

# ex: hammer_01.jpg -> 'hammer'
def label_from_filename(path):
    name = path.stem.lower()
    return re.sub(r"[\s_\-()\d]+$", "", name)

class ToolImageDataset(Dataset):
    # lists every .jpg in the folder and computes a label for each file
    def __init__(self, root, transform=None, target_transform=None,
                 class_to_idx=None, label_fn=label_from_filename):
        self.transform = transform
        self.target_transform = target_transform

        paths = [p for p in sorted(Path(root).iterdir())
                 if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("._")]
        if not paths:
            raise RuntimeError(f"No images found in {root}")

        labels = [label_fn(p) for p in paths]
        if class_to_idx is None:
            class_to_idx = {name: i for i, name in enumerate(sorted(set(labels)))}
        self.class_to_idx = class_to_idx
        self.class_names = sorted(class_to_idx, key=class_to_idx.get)

        # raises an error if any label isn't in the list 
        unknown = set(labels) - set(class_to_idx)
        if unknown:
            raise RuntimeError(f"Labels not in the training classes: {sorted(unknown)}")
        self.samples = [(p, class_to_idx[lab]) for p, lab in zip(paths, labels)]

    # returns how many images there are
    def __len__(self):
        return len(self.samples)

    # loads an image at a given index, fixes its orientation, converts it to a tensor, and returns it with its label number
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Turns every image into a tensor with shape (3, 128, 128), and converts pixel values from 1-255 to [0, 1]
transform = v2.Compose([
    v2.ToImage(),
    v2.Grayscale(num_output_channels=1),
    v2.Resize((128, 128)),
    v2.ToDtype(torch.float32, scale=True),
])

# builds the dataset and prints how many images and classes it found
dataset = ToolImageDataset(DATA_DIR, transform=transform)
class_names = dataset.class_names
print(f"Found {len(dataset)} images in {len(class_names)} classes: {class_names}")

# Split into 80% training, 20% validation
n_val = int(round(len(dataset) * VAL_FRACTION))
n_train = len(dataset) - n_val
train_data, val_data = random_split(
    dataset, [n_train, n_val],
    generator = torch.Generator().manual_seed(SEED),
)
print(f"Training images: {len(train_data)}   Validation images: {len(val_data)}")

# counts how many images each class has and returns a dictionary of class name to count
def class_counts(subset):
    counts = Counter(dataset.samples[i][1] for i in subset.indices)
    return {class_names[k]: counts[k] for k in range(len(class_names))}

print("Train per class:", class_counts(train_data))
print("Val   per class:", class_counts(val_data))

# DataLoaders - give the images to the NN in batches 
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# pulls out one batch to check shapes (should be [32, 3, 128, 128])
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.shape}")
print(f"Labels batch shape:  {train_labels.shape}") 

# Shows a few training images to confirm labels line up
fig = plt.figure(figsize=(8, 8))
for i in range(min(9, len(train_features))):
    fig.add_subplot(3, 3, i + 1)
    plt.title(class_names[train_labels[i].item()])
    plt.axis("off")
    plt.imshow(train_features[i].squeeze(), cmap="gray") 
#plt.show()



'''
Neural Network Architecture
'''

class Classifier(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        # Input image:
        # 1 x 128 x 128 = 49,152 values (Grayscale)

        self.fc1 = nn.Linear(1 * 128 * 128, 256)
        self.fc2 = nn.Linear(256, 224)
        self.fc3 = nn.Linear(224, 192)
        self.fc4 = nn.Linear(192, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, num_classes)


    def forward(self, x):
        x = x.view(x.shape[0], -1)
        
        # Hidden layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        

        # Output layer
        x = F.log_softmax(self.fc6(x), dim=1)
        return x


# Create the neural network
model = Classifier(num_classes=len(class_names))
print(model)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_dataloader:
        optimizer.zero_grad()
        log_probs = model(images)
        loss = criterion(log_probs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    average_loss = running_loss / len(train_dataloader)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_dataloader:
            log_probs = model(images)
            predictions = torch.argmax(log_probs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Error: {average_loss:.4f} | "
        f"Validation Accuracy: {accuracy:.2f}%"
    )
# Print the architecture

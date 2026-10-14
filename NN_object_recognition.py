# EGR 425 - Object Recognition Project
# Experiment 5 - CNN with augmentation and dropout
# Trying a few changes to help with new pictures

print("PROGRAM STARTED", flush=True)

from collections import Counter
from pathlib import Path
import random
import re
import torch
from PIL import Image, ImageOps
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import v2
import matplotlib.pyplot as plt


DATA_DIR = Path(__file__).resolve().parent / "ObjectImagesTrain"
BATCH_SIZE = 32
VAL_FRACTION = 0.20
SEED = 42
EPOCHS = 10
IMAGE_EXTS = {".jpg"}

MODEL_FILE = Path(__file__).resolve().parent / "best_cnn_experiment5.pth"

# Keep the split and starting weights the same
random.seed(SEED)
torch.manual_seed(SEED)


# Gets the object name from the filename
def label_from_filename(path):
    name = path.stem.lower()
    return re.sub(r"[\s_\-()\d]+$", "", name)


class ToolImageDataset(Dataset):

    # Finds the pictures and gives each one a label
    def __init__(self, root, transform=None):

        self.transform = transform

        paths = [
            p for p in sorted(Path(root).iterdir())
            if p.suffix.lower() in IMAGE_EXTS
            and not p.name.startswith("._")
        ]

        if not paths:
            raise RuntimeError(
                "No JPG images found in ObjectImagesTrain"
            )

        labels = [label_from_filename(p) for p in paths]

        self.class_names = sorted(set(labels))

        class_to_idx = {
            name: i
            for i, name in enumerate(self.class_names)
        }

        self.samples = [
            (p, class_to_idx[label])
            for p, label in zip(paths, labels)
        ]

    # Number of pictures
    def __len__(self):
        return len(self.samples)

    # Loads one picture and its correct label
    def __getitem__(self, index):

        path, label = self.samples[index]

        with Image.open(path) as picture:
            image = ImageOps.exif_transpose(
                picture
            ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


'''
Image Transforms
'''

# Validation pictures are not randomly changed
normal_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((128, 128)),
    v2.ToDtype(torch.float32, scale=True),
])


# Only training pictures get these small changes
train_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((128, 128)),
    v2.RandomRotation(15),
    v2.RandomAffine(
        degrees=0,
        translate=(0.06, 0.06)
    ),
    v2.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    v2.ToDtype(torch.float32, scale=True),
])


'''
Neural Network Architecture
'''

class Classifier(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        # Same three CNN layers as our best model
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # Makes the pictures smaller
        self.pool = nn.MaxPool2d(2, 2)

        # These layers decide which object it is
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Drop some neurons while training
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):

        # CNN layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten before the regular layers
        x = x.view(x.shape[0], -1)

        # Hidden layer with dropout
        x = self.dropout(F.relu(self.fc1(x)))

        # Output layer
        x = F.log_softmax(self.fc2(x), dim=1)

        return x


'''
Check Accuracy and Loss
'''

def check_results(model, loader, device, criterion):

    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            log_probs = model(images)

            loss = criterion(log_probs, labels)

            predictions = torch.argmax(
                log_probs, dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            loss_sum += loss.item() * labels.size(0)

    accuracy = 100 * correct / total
    average_loss = loss_sum / total

    return accuracy, average_loss


'''
Check Which Pictures Were Wrong
'''

def show_mistakes(model, dataset, indices, device):

    model.eval()

    correct = [0] * len(dataset.class_names)
    total = [0] * len(dataset.class_names)

    mistakes = []

    with torch.no_grad():

        for index in indices:

            image, actual = dataset[index]

            log_probs = model(
                image.unsqueeze(0).to(device)
            )

            probabilities = torch.exp(log_probs[0])

            predicted = torch.argmax(
                probabilities
            ).item()

            confidence = (
                probabilities[predicted].item() * 100
            )

            total[actual] += 1

            if predicted == actual:
                correct[actual] += 1

            else:
                mistakes.append((
                    dataset.samples[index][0].name,
                    dataset.class_names[actual],
                    dataset.class_names[predicted],
                    confidence
                ))

    print()
    print("ACCURACY FOR EACH OBJECT")

    for i, name in enumerate(dataset.class_names):

        accuracy = 100 * correct[i] / total[i]

        print(
            f"{name}: {correct[i]}/{total[i]} "
            f"= {accuracy:.2f}%"
        )

    print()
    print("INCORRECT VALIDATION PICTURES")

    if not mistakes:
        print("No mistakes on the validation pictures.")

    for filename, actual, predicted, confidence in mistakes:

        print(
            f"{filename} | "
            f"Actual: {actual} | "
            f"Predicted: {predicted} | "
            f"Confidence: {confidence:.2f}%"
        )


'''
Main Program
'''

def main():

    # Use GPU if available, otherwise CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using:", device)


    '''
    Load the Images
    '''

    normal_dataset = ToolImageDataset(
        DATA_DIR,
        normal_transform
    )

    augmented_dataset = ToolImageDataset(
        DATA_DIR,
        train_transform
    )

    class_names = normal_dataset.class_names

    print(
        f"Found {len(normal_dataset)} images "
        f"in {len(class_names)} classes: {class_names}"
    )


    '''
    Split the Data
    '''

    # Same balanced split as our earlier CNN tests
    random_generator = random.Random(SEED)

    train_indices = []
    val_indices = []

    for class_number in range(len(class_names)):

        indices = [
            i for i, sample in enumerate(normal_dataset.samples)
            if sample[1] == class_number
        ]

        random_generator.shuffle(indices)

        number_val = max(
            1, round(len(indices) * VAL_FRACTION)
        )

        val_indices.extend(indices[:number_val])
        train_indices.extend(indices[number_val:])


    # Training pictures get random changes
    train_data = Subset(
        augmented_dataset,
        train_indices
    )

    # Normal training pictures for checking accuracy
    train_check = Subset(
        normal_dataset,
        train_indices
    )

    # Validation pictures are never augmented
    val_data = Subset(
        normal_dataset,
        val_indices
    )


    '''
    DataLoaders
    '''

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    train_check_loader = DataLoader(
        train_check,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Training images:", len(train_indices))
    print("Validation images:", len(val_indices))

    print(
        "Train per class:",
        dict(Counter(
            normal_dataset.samples[i][1]
            for i in train_indices
        ))
    )


    '''
    Create the Network
    '''

    model = Classifier(
        len(class_names)
    ).to(device)

    print(model)

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # Lower the learning rate when validation loss
    # stops getting better
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4
    )


    '''
    Lists for Our Graphs
    '''

    train_accuracies = []
    val_accuracies = []

    train_losses = []
    val_losses = []

    best_loss = float("inf")
    best_accuracy = 0.0
    best_epoch = 0

    epochs_without_improvement = 0


    '''
    Train the Network
    '''

    for epoch in range(EPOCHS):

        # Training mode turns dropout on
        model.train()

        running_loss = 0.0
        total_images = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Clear old gradients
            optimizer.zero_grad()

            # Make predictions
            log_probs = model(images)

            # Find the error
            loss = criterion(log_probs, labels)

            # Backpropagation
            loss.backward()

            # Update the weights
            optimizer.step()

            running_loss += (
                loss.item() * labels.size(0)
            )

            total_images += labels.size(0)

        train_loss = running_loss / total_images


        # Check accuracy on normal training pictures
        train_accuracy, unused_loss = check_results(
            model,
            train_check_loader,
            device,
            criterion
        )

        # Check validation accuracy and loss
        val_accuracy, val_loss = check_results(
            model,
            val_loader,
            device,
            criterion
        )


        # Save results for our graphs
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)


        # Save the model with the lowest validation loss
        if val_loss < best_loss:

            best_loss = val_loss
            best_accuracy = val_accuracy
            best_epoch = epoch + 1

            epochs_without_improvement = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "epoch": best_epoch,
                "validation_accuracy": best_accuracy,
                "validation_loss": best_loss,
                "architecture": "CNN_RGB_16_32_64_dropout"
            }, MODEL_FILE)

        else:
            epochs_without_improvement += 1


        # Adjust learning rate if needed
        scheduler.step(val_loss)

        learning_rate = optimizer.param_groups[0]["lr"]


        # Print the results after each epoch
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Training Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f} | "
            f"Training Accuracy: {train_accuracy:.2f}% | "
            f"Validation Accuracy: {val_accuracy:.2f}% | "
            f"LR: {learning_rate:.6f}",
            flush=True
        )


        # Stop if validation loss has not improved
        if epochs_without_improvement >= 12:

            print(
                "Early stopping: validation loss "
                "stopped improving."
            )

            break


    '''
    Load the Best Model
    '''

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    best_train_accuracy, unused_loss = check_results(
        model,
        train_check_loader,
        device,
        criterion
    )


    '''
    Print Final Results
    '''

    print()
    print("TRAINING COMPLETE")

    print(
        "Best Saved Model Validation Accuracy:",
        round(best_accuracy, 2), "%"
    )

    print(
        "Best Saved Model Validation Loss:",
        round(best_loss, 4)
    )

    print("Best Epoch:", best_epoch)

    print(
        "Best Saved Model Training Accuracy:",
        round(best_train_accuracy, 2), "%"
    )

    print(
        "Highest Accuracy Seen During Training:",
        round(max(val_accuracies), 2), "%"
    )


    '''
    Check Incorrect Pictures
    '''

    show_mistakes(
        model,
        normal_dataset,
        val_indices,
        device
    )


    '''
    Graph Accuracy
    '''

    epoch_numbers = list(
        range(1, len(train_accuracies) + 1)
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        epoch_numbers,
        train_accuracies,
        label="Training Accuracy"
    )

    plt.plot(
        epoch_numbers,
        val_accuracies,
        label="Validation Accuracy"
    )

    plt.title("CNN Experiment 5 - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "cnn_experiment5_accuracy.png",
        dpi=300
    )

    plt.show()


    '''
    Graph Loss
    '''

    plt.figure(figsize=(9, 5))

    plt.plot(
        epoch_numbers,
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        epoch_numbers,
        val_losses,
        label="Validation Loss"
    )

    plt.title("CNN Experiment 5 - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "cnn_experiment5_loss.png",
        dpi=300
    )

    plt.show()


if __name__ == "__main__":
    main()

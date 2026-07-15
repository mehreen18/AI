# import matplotlib.pyplot as plt
# from torchvision import datasets, transforms

# transform = transforms.ToTensor()

# # CIFAR10 ek rangeen dataset hai (3 channels)
# data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)

# image, label = data[0]
# print("Poori image ka shape:", image.shape)   # [3, 32, 32]

# # Har channel ko alag alag dikhate hain
# fig, axes = plt.subplots(1, 4, figsize=(12, 3))

# axes[0].imshow(image.permute(1, 2, 0))     # asal rangeen image
# axes[0].set_title("Original (RGB)")

# axes[1].imshow(image[0], cmap='Reds')      # sirf Red channel
# axes[1].set_title("Red channel")

# axes[2].imshow(image[1], cmap='Greens')    # sirf Green channel
# axes[2].set_title("Green channel")

# axes[3].imshow(image[2], cmap='Blues')     # sirf Blue channel
# axes[3].set_title("Blue channel")

# for ax in axes:
#     ax.axis('off')

# plt.show()


"""
MNIST Digit Recognition - CNN (Convolutional Neural Network)
================================================================
Ye code handwritten digits (0-9) ko pehchanna seekhta hai.
Chalane ka tareeqa: python mnist_cnn.py
Pehli baar chalane par dataset khud download ho jayega (~10MB).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# 1. Data taiyar karna
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),                  # image -> numbers (0 se 1)
    transforms.Normalize((0.5,), (0.5,))    # numbers ko -1 se 1 tak scale karna
])

print("Dataset download/load ho raha hai...")
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

print(f"Training images: {len(train_data)}")
print(f"Testing images: {len(test_data)}")

# ---------------------------------------------------------
# 2. CNN Model
# ---------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))   # 14x14 -> 7x7
        x = x.view(x.size(0), -1)                  # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = CNN().to(device)
fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------
epochs = 5
print("\nTraining shuru ho rahi hai...\n")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    train_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] — Loss: {avg_loss:.4f} — Train Accuracy: {train_acc:.2f}%")

# ---------------------------------------------------------
# 4. Testing / Final Accuracy
# ---------------------------------------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nFinal Test Accuracy: {100 * correct / total:.2f}%")

# ---------------------------------------------------------
# 5. Model save karna
# ---------------------------------------------------------
torch.save(model.state_dict(), "mnist_cnn_model.pth")
print("\nModel save ho gaya: mnist_cnn_model.pth")

# ---------------------------------------------------------
# 6. Kuch sample predictions dikhana (terminal mein)
# ---------------------------------------------------------
print("\n--- Sample Predictions ---")
sample_images, sample_labels = next(iter(test_loader))
sample_images, sample_labels = sample_images.to(device), sample_labels.to(device)

with torch.no_grad():
    outputs = model(sample_images)
    _, predicted = torch.max(outputs, 1)

for i in range(10):
    status = "✓" if predicted[i] == sample_labels[i] else "✗"
    print(f"{status} Actual: {sample_labels[i].item()}   Predicted: {predicted[i].item()}")
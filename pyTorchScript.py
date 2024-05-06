import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pyTorchModel import CNN

# Set random seeds for reproducibility
torch.manual_seed(22)

# Define data transformations
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the training and testing data
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Define data loaders
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# Determine the shape of the input images
dataiter = iter(train_loader)
images, labels = next(dataiter)
img_shape = images[0].shape
print("Image Shape:", img_shape)

# Initialize the model
model = CNN()

# Print the model summary
print(model)

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 10
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()

# Testing the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        log_ps = model(images)
        _, predicted = torch.max(log_ps.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print('Accuracy: %.2f %%' % accuracy)

# Predict and show the first 20 images
images, labels = next(iter(test_loader))
model.eval()
with torch.no_grad():
    log_ps = model(images)
    ps = torch.exp(log_ps)
    _, predicted = torch.max(ps, 1)

plt.figure(figsize=(20, 9))
for i in range(20):
    plt.subplot(2, 10, i + 1)
    plt.imshow(images[i].numpy().squeeze(), cmap='gray')
    plt.axis('off')
    plt.title(f'Pred: {predicted[i].item()}, Actual: {labels[i].item()}', fontsize = 8)
plt.show()

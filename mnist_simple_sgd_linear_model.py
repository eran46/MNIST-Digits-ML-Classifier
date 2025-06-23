import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np


# Hyper Parameters
input_size = 784 # 28*28
num_classes = 10
num_epochs = 20
batch_size = 100
learning_rate = 1e-3
# 0.630 loss, 87%

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            download=True,
                            transform=transforms.ToTensor()
                            )

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor()
                           )

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

print(test_dataset)
# Number of datapoints: 10000

for images, labels in test_loader:
    print("Image batch shape:", images.shape)
    # Image batch shape: torch.Size([100, 1, 28, 28])

    print("Label batch shape:", labels.shape)
    # Label batch shape: torch.Size([100]) - as expected
    break


# Neural Network Simple Linear Model with 1 fully connected layer
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out


# initialize device (tested on apple silicone)
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

print(f"Using device: {device}")

net = Net(input_size, num_classes).to(device)
loss_func = nn.CrossEntropyLoss()   # loss
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)     # optimizer


# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch tensor shape (100, 1, 28, 28) to (100, 784)
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)

        net.train()     # good practice
        # forward
        yhat = net(images)

        # loss
        loss = loss_func(yhat, labels)

        print(f"on batch {i + 1} the loss was: {loss.item()}")

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

optimizer.zero_grad()

# Test the Model
correct = 0
total = 0
net.eval()  # good practice
with torch.inference_mode():
    for images, labels in test_loader:
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)

        total += labels.size(0)

        # inference
        yhat = net(images)

        pred_digits = torch.argmax(yhat, dim=1)     # yhat dim batch_size x 10, collapse dim 1.
        """
        # also possible:
        _, predicted = torch.max(yhat, dim=1)
        """

        correct_digits_tensor = (pred_digits == labels)
        correct_digits_tensor = correct_digits_tensor.to("cpu")
        correct += correct_digits_tensor.sum().item()


print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
torch.save(net.state_dict(), 'simple_sgd_linear_model.pth')

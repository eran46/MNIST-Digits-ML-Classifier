import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable


"""
# Hyper Parameters changed:
less epochs 12 -> 10
lower learning rate for adam optimizer vs sgd

optimizer changed:
now using adam
"""

# Hyper Parameters
input_size = 784 # 28*28
num_classes = 10
num_epochs = 10 # smaller
batch_size = 25 # smaller
learning_rate = 5e-4 # smaller
# adam, 0.257 loss, 92%

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out


# Initialize device
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

print(f"Using device: {device}")

net = Net(input_size, num_classes).to(device) # model

# Loss and Optimizer
criterion = nn.CrossEntropyLoss() # loss
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) # optimizer

epoch_mean_losses = []

# Train the Model
for epoch in range(num_epochs):
    losses = []
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28*28)).to(device)
        labels = Variable(labels).to(device)

        # Forward + Backward + Optimize
        net.train() # train mode
        yhat = net(images) # inference

        loss = criterion(yhat, labels) # loss
        losses.append(loss.item())

        loss.backward() # backward pass via pytorch dynamic graph
        optimizer.step() # GD step
        optimizer.zero_grad() # zero grads
    epoch_mean_loss = np.mean(losses)
    epoch_mean_losses.append(epoch_mean_loss)
    print(f"on epoch {epoch + 1} the mean loss was: {epoch_mean_loss}")


# Test the Model
correct = 0
total = 0
with torch.inference_mode(): # ADDED
    for images, labels in test_loader:
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)

        total += labels.size(0)

        net.eval()

        yhat = net(images)

        pred_digits = torch.argmax(yhat, dim=1) # yhat dim batch_size x 10, collapse dim 1.

        correct_digits_tensor = (pred_digits == labels)
        correct_digits_tensor = correct_digits_tensor.to("cpu")
        correct += correct_digits_tensor.sum().item()


print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
torch.save(net.state_dict(), 'adam_linear_model.pth')

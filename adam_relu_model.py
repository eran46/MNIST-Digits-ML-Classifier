import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable

"""
scheduled exponential decrease of lr on plateau at epoch ~25-30
larger batch size for stability but lower than initial 100
mean loss on last 10 epochs reaches sub 1e-5
epoch 49 sub 1e-6
1.45% error rate
"""

# CHANGE THIS
use_trained_weights = True


# Hyper Parameters
input_size = 784    # 28*28
hidden_size = 500
num_classes = 10
num_epochs = 50
batch_size = 50
learning_rate = 1e-3

# initialize device (tested on apple silicone)
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)
print(f"Using device: {device}")


# Neural Network Simple Linear Model with 1 fully connected layer
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


net = Net(input_size, num_classes).to(device)

# MNIST Dataset
test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor()
                           )

# Data Loader (Input Pipeline)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

if not use_trained_weights:
    train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                download=True,
                                transform=transforms.ToTensor()
                                )

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    loss_func = nn.CrossEntropyLoss()   # loss
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)     # optimizer

    # reduce LR if no improvement
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.33, threshold=2e-3
    )

    epoch_mean_losses = []

    # Train the Model
    for epoch in range(num_epochs):
        losses = []
        for i, (images, labels) in enumerate(train_loader):
            # Convert torch tensor to Variable
            images = Variable(images.view(-1, 28*28)).to(device)
            labels = Variable(labels).to(device)

            # Forward + Backward + Optimize
            net.train()     # train mode
            yhat = net(images)      # inference on batch

            loss = loss_func(yhat, labels)
            losses.append(loss.item())

            optimizer.zero_grad()  # zero grads
            loss.backward()  # backward pass via pytorch dynamic graph
            optimizer.step()  # GD step

        epoch_mean_loss = np.mean(losses)
        epoch_mean_losses.append(epoch_mean_loss.item())
        scheduler.step(epoch_mean_loss)  # adjusts LR based on mean batch-validation-loss (for each epoch)

        print(f"on epoch {epoch + 1} the mean loss was: {epoch_mean_loss}")

    # Save the Model
    torch.save(net.state_dict(), 'adam_relu_model.pth')

else:
    # load trained model
    net.load_state_dict(torch.load('adam_relu_model.pth'))

    # Test the Model
    correct = 0
    total = 0
    net.eval()  # good practice
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)

            total += labels.size(0)

            # inference
            yhat = net(images)

            pred_digits = torch.argmax(yhat, dim=1)  # yhat dim batch_size x 10, collapse dim 1.

            correct_digits_tensor = (pred_digits == labels)
            correct_digits_tensor = correct_digits_tensor.to("cpu")
            correct += correct_digits_tensor.sum().item()

    accuracy = 100 * (correct / total)
    print(f'Accuracy of the network on the 10000 test images: {accuracy}')

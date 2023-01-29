import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt

# Argument parser
parser = argparse.ArgumentParser(description='ECE361E HW1 - SimpleFC')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=25, help='Number of epoch to train')
# Define the learning rate of your optimizer
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
args = parser.parse_args()

# The size of input features
input_size = 28 * 28
# The number of target classes, you have 10 digits to classify
num_classes = 10

# Always make assignments to local variables from your args at the beginning of your code for better
# control and adaptability
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr

# Each experiment you will do will have slightly different results due to the randomness
# of the initialization value for the weights of the model. In order to have reproducible results,
# we have fixed a random seed to a specific value such that we "control" the randomness.
random_seed = 1
torch.manual_seed(random_seed)

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='data', train=False, transform=transforms.ToTensor())

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Define your model
class SimpleFC(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleFC, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, num_classes)

    # Your model only contains a single linear layer
    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        out = self.linear4(out)
        return out


model = SimpleFC(input_size, num_classes)

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

import torch.nn.functional as F

#LOSS PLOT
loss_plot_train = []
loss_plot_test = []
total_loss_epoch_train = 0
avg_loss_epoch_train = 0


#ACCURACY PLOT
acc_plot_train = []
acc_plot_test = []
total_acc_epoch_train = 0
avg_acc_epoch_train = 0




# Training loop
for epoch in range(num_epochs):
    # Training phase
    train_correct = 0
    train_total = 0
    train_loss = 0
    # Sets the model in training mode.
    model = model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Here we vectorize the 28*28 images as several 784-dimensional inputs
        images = images.view(-1, input_size)
        # Sets the gradients to zero
        optimizer.zero_grad()
        # The actual inference
        outputs = model(images)
        # Compute the loss between the predictions (outputs) and the ground-truth labels
        loss = criterion(outputs, labels)
        # Do backpropagation to update the parameters of your model
        loss.backward()
        # Performs a single optimization step (parameter update)
        optimizer.step()
        train_loss += loss.item()
        # The outputs are one-hot labels, we need to find the actual predicted
        # labels which have the highest output confidence
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        total_loss_epoch_train += (train_loss/(batch_idx + 1))
        total_acc_epoch_train += 100. * train_correct / train_total
        # Print every 100 steps the following information
        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                             len(train_dataset) // batch_size,
                                                                             train_loss / (batch_idx + 1),
                                                                             100. * train_correct / train_total))
    avg_loss_epoch_train = total_loss_epoch_train/(batch_idx + 1)
    print(avg_loss_epoch_train)
    loss_plot_train.append(avg_loss_epoch_train)
    total_loss_epoch_train = 0


    avg_acc_epoch_train = total_acc_epoch_train/(batch_idx + 1)
    print(avg_acc_epoch_train)
    acc_plot_train.append(avg_acc_epoch_train)
    total_acc_epoch_train = 0


    # Testing phase
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Here we vectorize the 28*28 images as several 784-dimensional inputs
            images = images.view(-1, input_size)
            # Perform the actual inference
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    print('Test accuracy: %.2f %% Test loss: %.4f' % (100. * test_correct / test_total, test_loss / (batch_idx + 1)))
    loss_plot_test.append(test_loss / (batch_idx + 1))
    acc_plot_test.append((100. * test_correct / test_total))

print(loss_plot_train)
print(loss_plot_test)

print(acc_plot_train)
print(acc_plot_test)


plt.title("Loss Plot", size = 25)
plt.xlabel("Epochs", size = 25)
plt.ylabel("Loss", size = 25)
plt.ylim(-0.01, 0.1)

epochs = list(range (1, num_epochs + 1))


plt.plot(epochs, loss_plot_train, label = "train")
plt.plot(epochs, loss_plot_test, label = "test")

plt.legend()

plt.grid()
plt.show()

plt.title("Accuracy Plot", size = 25)
plt.xlabel("Epochs", size = 25)
plt.ylabel("Accuracy", size = 25)

epochs = list(range (1, num_epochs+1))

plt.plot(epochs, acc_plot_train, label = "train")
plt.plot(epochs, acc_plot_test, label = "test")

plt.legend()

plt.grid()
plt.show()


#The model is overfitting because the accuracy is better on training data
# in comparison to test data. This observation reflects
# that the model is not performing relatively accurate predictions on its own.


#P2_Q2

#DROPOUT PROBABILITY = 0.0
# Define your model
class SimpleFC(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleFC, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.0)

    # Your model only contains a single linear layer
    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        out = self.dropout(out)
        #out = self.linear4(out)
        return out


model = SimpleFC(input_size, num_classes)

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

#LOSS PLOT
loss_plot_train = []
loss_plot_test = []
total_loss_epoch_train = 0
avg_loss_epoch_train = 0


# Training loop
for epoch in range(num_epochs):
    # Training phase
    train_correct = 0
    train_total = 0
    train_loss = 0
    # Sets the model in training mode.
    model = model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Here we vectorize the 28*28 images as several 784-dimensional inputs
        images = images.view(-1, input_size)
        # Sets the gradients to zero
        optimizer.zero_grad()
        # The actual inference
        outputs = model(images)
        # Compute the loss between the predictions (outputs) and the ground-truth labels
        loss = criterion(outputs, labels)
        # Do backpropagation to update the parameters of your model
        loss.backward()
        # Performs a single optimization step (parameter update)
        optimizer.step()
        train_loss += loss.item()
        # The outputs are one-hot labels, we need to find the actual predicted
        # labels which have the highest output confidence
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        total_loss_epoch_train += (train_loss/(batch_idx + 1))
        total_acc_epoch_train += 100. * train_correct / train_total
        # Print every 100 steps the following information
        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                             len(train_dataset) // batch_size,
                                                                             train_loss / (batch_idx + 1),
                                                                             100. * train_correct / train_total))
    avg_loss_epoch_train = total_loss_epoch_train/(batch_idx + 1)
    print(avg_loss_epoch_train)
    loss_plot_train.append(avg_loss_epoch_train)
    total_loss_epoch_train = 0


    # Testing phase
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Here we vectorize the 28*28 images as several 784-dimensional inputs
            images = images.view(-1, input_size)
            # Perform the actual inference
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    print('Test accuracy: %.2f %% Test loss: %.4f' % (100. * test_correct / test_total, test_loss / (batch_idx + 1)))
    loss_plot_test.append(test_loss / (batch_idx + 1))

print(loss_plot_train)
print(loss_plot_test)


plt.title("Loss Plot", size = 25)
plt.xlabel("Epochs", size = 25)
plt.ylabel("Loss", size = 25)
plt.ylim(-0.01, 3)

epochs = list(range (1, num_epochs + 1))


plt.plot(epochs, loss_plot_train, label = "train")
plt.plot(epochs, loss_plot_test, label = "test")

plt.legend()

plt.grid()
plt.show()

#here we evaluate the avg difference between train and test values

leng1 = len(loss_plot_train)
leng2 = len(loss_plot_test)
total1 = 0
total2 = 0
avg1 = 0
avg2 = 0

diff = 0

for i in range(leng1):
  total1+=loss_plot_train[i]
avg1 = total1/leng1


for j in range(leng2):
  total2+=loss_plot_test[j]
avg2 = total2/leng2

diff = abs(avg1 - avg2)


print("The avg diff between train and test values for a dropout probability of 0.0 is", diff)


#DROPOUT PROBABILITY = 0.2

# Define your model
class SimpleFC(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleFC, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)

    # Your model only contains a single linear layer
    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        out = self.dropout(out)
        #out = self.linear4(out)
        return out


model = SimpleFC(input_size, num_classes)

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

#LOSS PLOT
loss_plot_train = []
loss_plot_test = []
total_loss_epoch_train = 0
avg_loss_epoch_train = 0


# Training loop
for epoch in range(num_epochs):
    # Training phase
    train_correct = 0
    train_total = 0
    train_loss = 0
    # Sets the model in training mode.
    model = model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Here we vectorize the 28*28 images as several 784-dimensional inputs
        images = images.view(-1, input_size)
        # Sets the gradients to zero
        optimizer.zero_grad()
        # The actual inference
        outputs = model(images)
        # Compute the loss between the predictions (outputs) and the ground-truth labels
        loss = criterion(outputs, labels)
        # Do backpropagation to update the parameters of your model
        loss.backward()
        # Performs a single optimization step (parameter update)
        optimizer.step()
        train_loss += loss.item()
        # The outputs are one-hot labels, we need to find the actual predicted
        # labels which have the highest output confidence
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        total_loss_epoch_train += (train_loss/(batch_idx + 1))
        total_acc_epoch_train += 100. * train_correct / train_total
        # Print every 100 steps the following information
        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                             len(train_dataset) // batch_size,
                                                                             train_loss / (batch_idx + 1),
                                                                             100. * train_correct / train_total))
    avg_loss_epoch_train = total_loss_epoch_train/(batch_idx + 1)
    print(avg_loss_epoch_train)
    loss_plot_train.append(avg_loss_epoch_train)
    total_loss_epoch_train = 0


    # Testing phase
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Here we vectorize the 28*28 images as several 784-dimensional inputs
            images = images.view(-1, input_size)
            # Perform the actual inference
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    print('Test accuracy: %.2f %% Test loss: %.4f' % (100. * test_correct / test_total, test_loss / (batch_idx + 1)))
    loss_plot_test.append(test_loss / (batch_idx + 1))

print(loss_plot_train)
print(loss_plot_test)


plt.title("Loss Plot", size = 25)
plt.xlabel("Epochs", size = 25)
plt.ylabel("Loss", size = 25)
plt.ylim(0, 3)

epochs = list(range (1, num_epochs + 1))


plt.plot(epochs, loss_plot_train, label = "train")
plt.plot(epochs, loss_plot_test, label = "test")

plt.legend()

plt.grid()
plt.show()

#here we evaluate the avg difference between train and test values

leng1 = len(loss_plot_train)
leng2 = len(loss_plot_test)
total1 = 0
total2 = 0
avg1 = 0
avg2 = 0

diff = 0

for i in range(leng1):
  total1+=loss_plot_train[i]
avg1 = total1/leng1


for j in range(leng2):
  total2+=loss_plot_test[j]
avg2 = total2/leng2

diff = abs(avg1 - avg2)


print("The avg diff between train and test values for a dropout probability of 0.2 is", diff)


#DROPOUT PROB = 0.5

# Define your model
class SimpleFC(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleFC, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    # Your model only contains a single linear layer
    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        out = self.dropout(out)
        #out = self.linear4(out)
        return out


model = SimpleFC(input_size, num_classes)

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

#LOSS PLOT
loss_plot_train = []
loss_plot_test = []
total_loss_epoch_train = 0
avg_loss_epoch_train = 0


# Training loop
for epoch in range(num_epochs):
    # Training phase
    train_correct = 0
    train_total = 0
    train_loss = 0
    # Sets the model in training mode.
    model = model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Here we vectorize the 28*28 images as several 784-dimensional inputs
        images = images.view(-1, input_size)
        # Sets the gradients to zero
        optimizer.zero_grad()
        # The actual inference
        outputs = model(images)
        # Compute the loss between the predictions (outputs) and the ground-truth labels
        loss = criterion(outputs, labels)
        # Do backpropagation to update the parameters of your model
        loss.backward()
        # Performs a single optimization step (parameter update)
        optimizer.step()
        train_loss += loss.item()
        # The outputs are one-hot labels, we need to find the actual predicted
        # labels which have the highest output confidence
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        total_loss_epoch_train += (train_loss/(batch_idx + 1))
        total_acc_epoch_train += 100. * train_correct / train_total
        # Print every 100 steps the following information
        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                             len(train_dataset) // batch_size,
                                                                             train_loss / (batch_idx + 1),
                                                                             100. * train_correct / train_total))
    avg_loss_epoch_train = total_loss_epoch_train/(batch_idx + 1)
    print(avg_loss_epoch_train)
    loss_plot_train.append(avg_loss_epoch_train)
    total_loss_epoch_train = 0


    # Testing phase
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Here we vectorize the 28*28 images as several 784-dimensional inputs
            images = images.view(-1, input_size)
            # Perform the actual inference
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    print('Test accuracy: %.2f %% Test loss: %.4f' % (100. * test_correct / test_total, test_loss / (batch_idx + 1)))
    loss_plot_test.append(test_loss / (batch_idx + 1))

print(loss_plot_train)
print(loss_plot_test)


plt.title("Loss Plot", size = 25)
plt.xlabel("Epochs", size = 25)
plt.ylabel("Loss", size = 25)
plt.ylim(0, 3)

epochs = list(range (1, num_epochs + 1))


plt.plot(epochs, loss_plot_train, label = "train")
plt.plot(epochs, loss_plot_test, label = "test")

plt.legend()

plt.grid()
plt.show()

#here we evaluate the avg difference between train and test values

leng1 = len(loss_plot_train)
leng2 = len(loss_plot_test)
total1 = 0
total2 = 0
avg1 = 0
avg2 = 0

diff = 0

for i in range(leng1):
  total1+=loss_plot_train[i]
avg1 = total1/leng1


for j in range(leng2):
  total2+=loss_plot_test[j]
avg2 = total2/leng2

diff = abs(avg1 - avg2)


print("The avg diff between train and test values for a dropout probability of 0.5 is", diff)



#DROPOUT PROB = 0.8

# Define your model
class SimpleFC(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleFC, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.8)

    # Your model only contains a single linear layer
    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        out = self.dropout(out)
        #out = self.linear4(out)
        return out


model = SimpleFC(input_size, num_classes)

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

#LOSS PLOT
loss_plot_train = []
loss_plot_test = []
total_loss_epoch_train = 0
avg_loss_epoch_train = 0


# Training loop
for epoch in range(num_epochs):
    # Training phase
    train_correct = 0
    train_total = 0
    train_loss = 0
    # Sets the model in training mode.
    model = model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Here we vectorize the 28*28 images as several 784-dimensional inputs
        images = images.view(-1, input_size)
        # Sets the gradients to zero
        optimizer.zero_grad()
        # The actual inference
        outputs = model(images)
        # Compute the loss between the predictions (outputs) and the ground-truth labels
        loss = criterion(outputs, labels)
        # Do backpropagation to update the parameters of your model
        loss.backward()
        # Performs a single optimization step (parameter update)
        optimizer.step()
        train_loss += loss.item()
        # The outputs are one-hot labels, we need to find the actual predicted
        # labels which have the highest output confidence
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        total_loss_epoch_train += (train_loss/(batch_idx + 1))
        total_acc_epoch_train += 100. * train_correct / train_total
        # Print every 100 steps the following information
        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                             len(train_dataset) // batch_size,
                                                                             train_loss / (batch_idx + 1),
                                                                             100. * train_correct / train_total))
    avg_loss_epoch_train = total_loss_epoch_train/(batch_idx + 1)
    print(avg_loss_epoch_train)
    loss_plot_train.append(avg_loss_epoch_train)
    total_loss_epoch_train = 0


    # Testing phase
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Here we vectorize the 28*28 images as several 784-dimensional inputs
            images = images.view(-1, input_size)
            # Perform the actual inference
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    print('Test accuracy: %.2f %% Test loss: %.4f' % (100. * test_correct / test_total, test_loss / (batch_idx + 1)))
    loss_plot_test.append(test_loss / (batch_idx + 1))

print(loss_plot_train)
print(loss_plot_test)


plt.title("Loss Plot", size = 25)
plt.xlabel("Epochs", size = 25)
plt.ylabel("Loss", size = 25)
plt.ylim(2, 5)

epochs = list(range (1, num_epochs + 1))


plt.plot(epochs, loss_plot_train, label = "train")
plt.plot(epochs, loss_plot_test, label = "test")

plt.legend()

plt.grid()
plt.show()

#here we evaluate the avg difference between train and test values

leng1 = len(loss_plot_train)
leng2 = len(loss_plot_test)
total1 = 0
total2 = 0
avg1 = 0
avg2 = 0

diff = 0

for i in range(leng1):
  total1+=loss_plot_train[i]
avg1 = total1/leng1


for j in range(leng2):
  total2+=loss_plot_test[j]
avg2 = total2/leng2

diff = abs(avg1 - avg2)


print("The avg diff between train and test values for a dropout probability of 0.8 is", diff)

# The best results came from a dropout probability of 0.0
# (w/ an average diff of 0.05) and the worst results came from a
# dropout probability of 0.5 (w/ an average diff of 1.95).

# Although dropout is supposed to regularize and prevent overfitting,
# it is possible that it made this issue worse because it was placed right
# before the last layer so it has not as much time to correct errors
# before classification. The network may also be smaller in comparison
# to the dataset.



#P2Q3


#running 0.0 dropout probability...

# Define your model
class SimpleFC(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleFC, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.0)

    # Your model only contains a single linear layer
    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        out = self.dropout(out)
        #out = self.linear4(out)
        return out


model = SimpleFC(input_size, num_classes)

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

#LOSS PLOT
loss_plot_train = []
loss_plot_test = []
total_loss_epoch_train = 0
avg_loss_epoch_train = 0


#ACCURACY PLOT
acc_plot_train = []
acc_plot_test = []
total_acc_epoch_train = 0
avg_acc_epoch_train = 0


# Training loop
for epoch in range(num_epochs):
    # Training phase
    train_correct = 0
    train_total = 0
    train_loss = 0
    # Sets the model in training mode.
    model = model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Here we vectorize the 28*28 images as several 784-dimensional inputs
        images = images.view(-1, input_size)
        # Sets the gradients to zero
        optimizer.zero_grad()
        # The actual inference
        outputs = model(images)
        # Compute the loss between the predictions (outputs) and the ground-truth labels
        loss = criterion(outputs, labels)
        # Do backpropagation to update the parameters of your model
        loss.backward()
        # Performs a single optimization step (parameter update)
        optimizer.step()
        train_loss += loss.item()
        # The outputs are one-hot labels, we need to find the actual predicted
        # labels which have the highest output confidence
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        total_loss_epoch_train += (train_loss/(batch_idx + 1))
        total_acc_epoch_train += 100. * train_correct / train_total
        # Print every 100 steps the following information
        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                             len(train_dataset) // batch_size,
                                                                             train_loss / (batch_idx + 1),
                                                                             100. * train_correct / train_total))
    avg_loss_epoch_train = total_loss_epoch_train/(batch_idx + 1)
    print(avg_loss_epoch_train)
    loss_plot_train.append(avg_loss_epoch_train)
    total_loss_epoch_train = 0

    avg_acc_epoch_train = total_acc_epoch_train/(batch_idx + 1)
    print(avg_acc_epoch_train)
    acc_plot_train.append(avg_acc_epoch_train)
    total_acc_epoch_train = 0


    # Testing phase
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Here we vectorize the 28*28 images as several 784-dimensional inputs
            images = images.view(-1, input_size)
            # Perform the actual inference
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    print('Test accuracy: %.2f %% Test loss: %.4f' % (100. * test_correct / test_total, test_loss / (batch_idx + 1)))
    loss_plot_test.append(test_loss / (batch_idx + 1))
    acc_plot_test.append((100. * test_correct / test_total))

print(loss_plot_train)
print(loss_plot_test)

print(acc_plot_train)
print(acc_plot_test)

tot = 0
tr_avg = 0
te_avg = 0
for i in range(len(acc_plot_train)):
  tot += acc_plot_train[i]

tr_avg = tot/len(acc_plot_train)

tot = 0
for j in range(len(acc_plot_test)):
  tot += acc_plot_test[j]

te_avg = tot/len(acc_plot_test)

print(tr_avg)
print(te_avg)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root='data', train=True, transform=transform, download=True)
test_dataset = dsets.MNIST(root='data', train=False, transform=transform)

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define your model
class SimpleFC(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleFC, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.0)

    # Your model only contains a single linear layer
    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        out = self.dropout(out)
        #out = self.linear4(out)
        return out


model = SimpleFC(input_size, num_classes)

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

#LOSS PLOT
loss_plot_train = []
loss_plot_test = []
total_loss_epoch_train = 0
avg_loss_epoch_train = 0


#ACCURACY PLOT
acc_plot_train = []
acc_plot_test = []
total_acc_epoch_train = 0
avg_acc_epoch_train = 0


# Training loop
for epoch in range(num_epochs):
    # Training phase
    train_correct = 0
    train_total = 0
    train_loss = 0
    # Sets the model in training mode.
    model = model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Here we vectorize the 28*28 images as several 784-dimensional inputs
        images = images.view(-1, input_size)
        # Sets the gradients to zero
        optimizer.zero_grad()
        # The actual inference
        outputs = model(images)
        # Compute the loss between the predictions (outputs) and the ground-truth labels
        loss = criterion(outputs, labels)
        # Do backpropagation to update the parameters of your model
        loss.backward()
        # Performs a single optimization step (parameter update)
        optimizer.step()
        train_loss += loss.item()
        # The outputs are one-hot labels, we need to find the actual predicted
        # labels which have the highest output confidence
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        total_loss_epoch_train += (train_loss/(batch_idx + 1))
        total_acc_epoch_train += 100. * train_correct / train_total
        # Print every 100 steps the following information
        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                             len(train_dataset) // batch_size,
                                                                             train_loss / (batch_idx + 1),
                                                                             100. * train_correct / train_total))
    avg_loss_epoch_train = total_loss_epoch_train/(batch_idx + 1)
    print(avg_loss_epoch_train)
    loss_plot_train.append(avg_loss_epoch_train)
    total_loss_epoch_train = 0

    avg_acc_epoch_train = total_acc_epoch_train/(batch_idx + 1)
    print(avg_acc_epoch_train)
    acc_plot_train.append(avg_acc_epoch_train)
    total_acc_epoch_train = 0


    # Testing phase
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Here we vectorize the 28*28 images as several 784-dimensional inputs
            images = images.view(-1, input_size)
            # Perform the actual inference
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    print('Test accuracy: %.2f %% Test loss: %.4f' % (100. * test_correct / test_total, test_loss / (batch_idx + 1)))
    loss_plot_test.append(test_loss / (batch_idx + 1))
    acc_plot_test.append((100. * test_correct / test_total))

print(loss_plot_train)
print(loss_plot_test)

print(acc_plot_train)
print(acc_plot_test)

tot = 0
tr_avg = 0
te_avg = 0
for i in range(len(acc_plot_train)):
  tot += acc_plot_train[i]

tr_avg = tot/len(acc_plot_train)

tot = 0
for j in range(len(acc_plot_test)):
  tot += acc_plot_test[j]

te_avg = tot/len(acc_plot_test)

print(tr_avg)
print(te_avg)


# The normalized experiments yielded better accuracy
# results but the total time for training was much higher.
# It may be worth the tradeoff for larger datasets but for this one,
# it is negotiable since the accuracy was high with and without
# normalization. Another observation to note is that the first
# epoch when the model reached 96% training accuracy for
# normalization was 3/25 while unnormalized was 5/25.
# The accuracy in general was better but the time took longer for training.

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import argparse
import time
from thop import profile
from torchsummary import summary
from models import vgg11_pt, vgg16_pt, mobilenet_pt
import matplotlib.pyplot as plt 

# Argument parser
parser = argparse.ArgumentParser(description='ECE361E HW3 - Starter PyTorch code')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
args = parser.parse_args()

# Always make assignments to local variables from your args at the beginning of your code for better
# control and adaptability
num_epochs = args.epochs
batch_size = args.batch_size

# Each experiment you will do will have slightly different results due to the randomness
# of the initialization value for the weights of the model. In order to have reproducible results,
# we have fixed a random seed to a specific value such that we "control" the randomness.
random_seed = 1
torch.manual_seed(random_seed)

# CIFAR10 Dataset (Images and Labels)
train_dataset = dsets.CIFAR10(root='data', train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]), download=True)

test_dataset = dsets.CIFAR10(root='data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]))

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# TODO: Get VGG11 model
model = vgg16_pt.VGG16()
# TODO: Put the model on the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device) 

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters())

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

# Training loop
total_training_time = 0

# Training loop
for epoch in range(num_epochs):
    start = time.time()
    # Training phase
    train_correct = 0
    train_total = 0
    train_loss = 0
    # Sets the model in training mode.
    model = model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        # TODO: Put the images and labels on the GPU
        images, labels = images.to(device), labels.to(device)

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
        # Print every 100 steps the following information
        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                             len(train_dataset) // batch_size,
                                                                             train_loss / (batch_idx + 1),
                                                                             100. * train_correct / train_total))
    end = time.time()
    per_epoch_training_time = end - start 
    total_training_time += per_epoch_training_time

    train_loss_list.append(train_loss / (batch_idx + 1))
    train_acc_list.append(100. * train_correct / train_total)

    # Testing phase
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    total_inference_time = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # TODO: Put the images and labels on the GPU
            images, labels = images.to(device), labels.to(device)

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
    print('Test loss: %.4f Test accuracy: %.2f %%' % (test_loss / (batch_idx + 1),100. * test_correct / test_total))

    test_loss_list.append(test_loss / (batch_idx + 1))
    test_acc_list.append(100. * test_correct / test_total)

    # TODO: Save the PyTorch model in .pt format
    torch.save(model.state_dict(), 'results/16saved_in_epoch_loop.pt')
# solution
torch.save(model.state_dict(), 'results/vgg-16_results.pt')

#sample_input = torch.randn(1, 3, 32, 32)
#sample_input = sample_input.to(device)

macs, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).to(device), ))



# quantitative results needed for the table 
print('Overall training accuracy at the end of %d epochs : %.4f percent' % (num_epochs, train_acc_list[-1])) 
print('Overall test accuracy at the end of %d epochs : %.4f percent' % (num_epochs, test_acc_list[-1]))
print('Total time for training : %.4f seconds' % (total_training_time))


# quantitative results needed for the table
print('Model statistics from thop') 
print('Number of macs : %d' % (macs)) # for compressed cnn 93296
print('Number of flops : %d' % (macs * 2)) # for compressed cnn 186592
print('Model params : %d' % (params)) # for compressed cnn 4266  
print('Model summary from torch summary')
print(summary(model, (3, 32, 32)))

print(test_acc_list)
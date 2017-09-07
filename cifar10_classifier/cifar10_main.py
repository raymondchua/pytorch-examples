import torch
import torchvision
import torchvision.transforms as transforms

# import matplotlib.pyplot as plt
import numpy as np

from cifar10_net import Net

import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

transform = transforms.Compose(
	[transforms.ToTensor(), #convert rgb image to float [0.0, 1.0]
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #Given mean and std. dev, normalize each channel

#download the cifar10 training dataset and apply the transform
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

#load the training set, data reshuffled at every epoch , set batch size to 4 and num_workers determine
#how many subprocesses to use for data loading
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

#download and load the test set. Same transform applies for test set except that no reshuffling is applied
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

#define the classes of the cifar10 dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image
# def imshow(img):
# 	img = img / 2 + 0.5     # unnormalize
# 	npimg = img.numpy()
# 	plt.imshow(np.transpose(npimg, (1, 2, 0)))


# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
# plt.show()

net = Net()
net = net.cuda() #activate cuda for neural network

#use the cross entropy loss as our loss function and SGD as our optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

num_iter = 5

for epoch in range(num_iter):
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		 # get the inputs
		inputs, labels = data

		# wrap them in Variable
		inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda()) #send inputs and label to cuda too

		# zero the parameter gradients
		optimizer.zero_grad()

		 # forward + backward + optimize
		outputs = net(inputs)

		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.data[0]
		if i % 2000 == 1999:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' %
				(epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

print('Finished Training')

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
# imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
# plt.show()

outputs = net(Variable(images.cuda()))
_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# for data in testloader:
# 	images, labels = data
# 	outputs = net(Variable(images.cuda()))
# 	_, predicted = torch.max(outputs.data, 1)
# 	c = (predicted == labels).squeeze()
# 	for i in range(4):
# 		label = labels[i]
# 		class_correct[label] += c[i]
# 		class_total[label] += 1


# for i in range(10):
# 	print('Accuracy of %5s : %2d %%' % (
# 		classes[i], 100 * class_correct[i] / class_total[i]))




import torch
import torchvision
import torchvision.transforms as transforms


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



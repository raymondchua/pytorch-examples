from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5) #5x5 kernel with input channel 3 and output channel 6
		self.pool = nn.MaxPool2d(2, 2) #2x2 max pooling with stride 2
		self.conv2 = nn.Conv2d(6, 16, 5) #5x5 kernel with input channel 6 and output channel 16
		self.fc1 = nn.Linear(16 * 5 * 5, 120) #applies linear transformation such at y=Ax+b, (16*5*5) input channel, 120 output channels
		self.fc2 = nn.Linear(120, 84) #applies linear transformation such at y=Ax+b, 120 input channels, 84 output channels
		self.fc3 = nn.Linear(84, 10) #applies linear transformation such at y=Ax+b, 84 input channels, 10 output channels

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x))) #2d conv -> relu -> max pooling
		x = self.pool(F.relu(self.conv2(x))) #2d conv -> relu -> max pooling
		x = x.view(-1, 16 * 5 * 5) #returns same data but different size, looks like tf.reshape
		x = F.relu(self.fc1(x)) #fully-connected -> relu
		x = F.relu(self.fc2(x)) #fully-connected -> relu
		x = self.fc3(x) #fully-connected
		return x


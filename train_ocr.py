import torch
import os
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import skimage.io
#from q4 import *
import skimage.transform
import matplotlib.pyplot as plt


num_epochs = 6
batch_size = 20
learning_rate = 0.01

train_dataset = dsets.EMNIST(root='.',
                             split= 'balanced',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = pad_value = kwargs.get('padder', 255)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

class Net(nn.Module):
    def __init__(self):
        super (Net, self).__init__ ()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 47)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "Net"

model = Net()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for t in range(num_epochs):
    total_loss = 0
    total = 0
    correct = 0
    for i, data in enumerate(train_loader, 0):
        #print(i)
        inputs, labels = data
        #inp = inputs.numpy()
        # plt.imshow(inp[0,0,:,:])
        # plt.show()
        inputs = Variable(inputs)
        labels = Variable(labels)
        optimizer.zero_grad()
        probs = model(inputs)
        loss = criterion(probs, labels)
        _, predicted = torch.max(probs.data, dim=1)
        total += labels.size (0)
        correct += torch.sum(labels == predicted)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('Loss',total_loss,'Acc', correct.item()/total)
torch.save(model,'model.pt')
#model = torch.load('model.pt')






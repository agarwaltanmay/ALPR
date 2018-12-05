import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import scipy.io

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

def train_model(dataloader, optimizer, model, max_iters=100, thresh=0.75):
    train_loss = []
    train_acc = []
    for itr in range(max_iters):
        total_loss = 0
        total_acc = 0
        batch_num = 0
        for i_batch, sample_batched in enumerate(dataloader):
            data, labels = sample_batched
            data = data.view(-1, 1, 32, 32)

            optimizer.zero_grad()
            probs = model(data)
            labels = labels.view(-1, 1)
            
            loss = F.mse_loss(probs, labels)
            loss.backward()
            optimizer.step()

            predicted_labels = (probs > thresh).float()
            total_acc += float((predicted_labels == labels).sum()) / batch_size
            total_loss += loss.item()
            batch_num += 1

        total_acc = total_acc / batch_num
        train_acc.append(total_acc)
        train_loss.append(total_loss / batch_num)
        if itr % 2 == 1:
            print("Epoch: {:02d} \t training_loss: {:.2f} \t training_accuracy : {:.2f}".format(
                itr+1, total_loss, total_acc))
    
    torch.save(model, './saved_model/trained_model.pt')
    return train_loss, train_acc


def test_model(dataloader, model, thresh=0.75):
    total_acc = 0
    batch_num = 0
    
    for i_batch, sample_batched in enumerate(dataloader):
        data, labels = sample_batched
        data = data.view(-1, 1, 32, 32)

        probs = model(data)
        labels = labels.view(-1, 1)

        predicted_labels = (probs > thresh).float()
        total_acc += float((predicted_labels == labels).sum()) / batch_size
        batch_num += 1

    total_acc = total_acc / batch_num
    test_acc = total_acc
    print("test_accuracy : {:.2f}".format(test_acc))
    return test_acc

if __name__ == "__main__":
    max_iters = 30
    batch_size = 30
    momentum = 0.9
    learning_rate = 0.001  # 0.005  # 0.05 #0.0005

    model = Net()

    train_x = np.load('./data/train_x.npy')
    size = int(train_x.shape[0] / 2)
    train_y = np.ones(size)
    train_y = np.hstack((train_y, np.zeros(size)))
    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).float()
    
    train = torch.utils.data.TensorDataset(train_x, train_y)
    nist_dataloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    train_loss, train_acc = train_model(nist_dataloader, optimizer, model, max_iters=max_iters)

    model = torch.load('./saved_model/trained_model.pt')

    test_x = np.load('./data/test_x.npy')
    size = int(test_x.shape[0] / 2)
    test_y = np.ones(size)
    test_y = np.hstack((test_y, np.zeros(size)))

    test_x = torch.from_numpy(test_x).float()
    test_y = torch.from_numpy(test_y).float()

    test = torch.utils.data.TensorDataset(test_x, test_y)
    test_nist_dataloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=True)

    test_acc = test_model(test_nist_dataloader, model, thresh=0.75)

    epochs = np.arange(1, max_iters + 1)
    plt.plot(epochs, train_acc, 'r-', label='Train Accuracy')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Plot')
    plt.show()

    plt.plot(epochs, train_loss, 'r-', label='Train Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.title('Loss Plot')
    plt.show()

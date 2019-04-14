import torch
import torchvision
import torchvision.transforms as tranforms
import numpy
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.functional as F
import torch.optim as optim


def download_data():
    transform = transforms.Compose([transforms.ToTensor(), tranforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.Dataloader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.Dataloader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'trunk')
    return trainloader, testloader


def show_data():
    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    trainloader, testloader = download_data()
    dataiter = iter(trainloader)
    images, label = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[label[j]] for j in range(4)))


class Net(nn.Module):
    def __init(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train():
    net = Net()
    criterion = nn.CrossEntropy()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    trainloader, testloader = download_data()
    
    # training
    for epoch in range(2):
        running_loss = 0.0
        for i, data = in enumerate(trainloader, 0)
            inputs, laebls = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
                running_loss = 0.0
    print('Finish training.')

    # testing
    corret = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            corret += (predicted == labels).sum().item()
    print('Acc %d %%' % 100 * corret / total)


if __name__ == '__main__':
    train()

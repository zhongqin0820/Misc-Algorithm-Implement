import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Moudule):
    def __init__(self):
        super(Net, self).__init__()
        # input chan, output chan, kernel square size: 5 x 5
        # accepts (sample_num, chan_num, height, width)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.f1 = nn.Linear(16 * 5 * 5, 120)
        self.f2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, f):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def num_flat_features(self, x):
        """
        :return int: the num of the features
        """
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
#return the params that the net defines
# params = list(net.parameters())
input = torch.randn(1, 1, 32, 32)
# gen target and reshape it
target = torch.randn(10).view(1, -1)
out = net(input)
# out.zero_grad()
# out.backward(torch.randn(1, 10))

# define loss
criterion = nn.MSELoss()
loss = criterion(out, target)

# backward 
net.zero_grad()
loss.backward()

# update weight (torch.optim)
optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()
out = net(input)
loss = criterion(out, target)
loss.backward()
optimizer.step()

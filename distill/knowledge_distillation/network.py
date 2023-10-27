import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def load_network(model, model_load, num_class, device):
    if model == '1':
        network = Student1(num_class)

    elif model == 'cnn':
        network = ModelCNN_MNIST(num_class)

    if model_load:
        check_point = torch.load(model_load, map_location=device)
        status = network.load_state_dict(check_point['state_dict'])
        print("Load model weights from %s: "%model_load, status)
    network = network.to(device)

    return network

class Student1(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ModelCNN_MNIST(nn.Module):
    def __init__(self, input_shape, nb_classes=10, *args, **kwargs):
        super(ModelCNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, nb_classes)

    def forward(self, x):
        x = x.view(-1, 1,28,28).float()
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,3*3*64 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
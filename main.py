
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

input_size = 784
hidden_size1 = 512
hidden_size2 = 256
hidden_size3 = 128
hidden_size4 = 64
num_classes = 10
num_epochs = 40
batch_size = 64
learning_rate = 0.001

test_dataset = torchvision.datasets.FashionMNIST(
     root = './data/FashionMNIST',
     train = False,
     download = True,
     transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485,),(0.229,))]))
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=len(test_dataset),
                                          shuffle=False)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1,hidden_size2,hidden_size3, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3) 
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, num_classes) 

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out

class ConvNet(nn.Module):
    def __init__(self,num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d((2, 2),stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d((2, 2),stride=2))
        self.fc1 = nn.Linear(7*7*64,num_classes)
    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.reshape(out.size(0), -1)
        #x = x.view(-1, 256)
        out = self.fc1(out)
        return out
        
        
model = NeuralNet(input_size,hidden_size1,hidden_size2,hidden_size3, num_classes)
model1 = ConvNet(num_classes)

PATH1 = "./basicnnfmnistmodel1.pt"
model.load_state_dict(torch.load(PATH1))
PATH2 = "./cnnfmnistmodel3.pt"
model1.load_state_dict(torch.load(PATH2))

model.eval()
correct = 0
total = 0
for images, labels in test_loader:
    images = images.reshape(-1, 28*28)
    labels = labels
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

model1.eval()
correct1 = 0
total1 = 0
for images, labels in test_loader:
    images = images
    labels = labels
    outputs1 = model1(images)
    _, predicted1 = torch.max(outputs1.data, 1)
    total1 += labels.size(0)
    correct1 += (predicted1 == labels).sum().item()

print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct1 / total1))

y_pred_mlnn = F.softmax(outputs, dim=1)
loss_m_l_nn = F.cross_entropy(y_pred_mlnn, labels)

y_pred_cnn = F.softmax(outputs1, dim=1)
loss_cnn = F.cross_entropy(y_pred_cnn, labels)

with open("multi-layer-net.txt", 'w') as f:
  f.write("Loss on Test Data : {}\n".format(loss_m_l_nn))
  f.write("Accuracy on Test Data : {}\n".format(100 * correct / total))
  f.write("gt_label,pred_label \n")
  for idx in range(len(labels)):
    f.write("{},{}\n".format(labels[idx], predicted[idx]))

with open("convolution-neural-net.txt", 'w') as f:
  f.write("Loss on Test Data : {}\n".format(loss_cnn))
  f.write("Accuracy on Test Data : {}\n".format(100 * correct1 / total))
  f.write("gt_label,pred_label \n")
  for idx in range(len(labels)):
    f.write("{},{}\n".format(labels[idx], predicted1[idx]))


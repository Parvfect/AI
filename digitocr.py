import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision 
from torchvision import transforms, datasets #has vision datsets 


train = datasets.MNIST("", 
train = True, 
download = True, 
transform = transforms.Compose([transforms.ToTensor()])) 



test = datasets.MNIST("", 
train = False, 
download = True, 
transform = transforms.Compose([transforms.ToTensor()])) 


trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)
testset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #Order in which this is defined is irrelevant
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64,10) #We have ten classes of numbers
    
    def forward(self, x):
        #We are defining a feet forward neural net
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #We do not want to run activa tion on the last layers
        x = self.fc4(x)

        #We want a probabilty distribution at the end. 
        return F.log_softmax(x, dim=1) 
        
    

net=Net()

     

import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr = 0.001)
EPOCHS = 3 #Number of times we pass through our data

correct = 0
total = 0

for epoch in range(EPOCHS):
    for data in trainset:
        #data is a batch of featuresets and labels
        X, y = data #Ten feature sets and labels - an image and a label
        net.zero_grad()
        #Resets
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y)
        loss.backward() #backpropogating the losses
        optimizer.step() #adjusting the weights using the optim
    print(loss)

with torch.no_grad(): #Testing the data we don't want to optimize it to this data
    for data in trainset:
        X,y = data
        output = net(X.view(-1,784))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct+=1
            total+=1

print("Accuracy ,", round(correct/total*100))

import matplotlib.pyplot as plt 
plt.imshow(X[0].view(28,28))
plt.show()
print(torch.argmax(net(X[0].view(-1,784))[0]))
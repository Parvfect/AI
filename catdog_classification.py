
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split

""" To do - Transfer learning to use this to identify some other image features"""

REBUILD_DATA = True
REBUILD_MODEL = True

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")

else:
    device = torch.device("cpu")
    print("Running on the CPU")

# Preprocessing step can take a lot of time - you want to run it as few as possible

class DogsvCats():
    """ Repeated steps basically means that encapsulating it is helpful """

    # Images needs to be of the same shape
    img_size = 50
    cats = "/home/parvfect/Documents/AI/catdog/PetImages/Cat"
    dogs = "/home/parvfect/Documents/AI/catdog/PetImages/Dog"
    Labels = {cats:0, dogs:1}

    training_data = []
    cat_count = 0
    dog_count = 0

    # Balance is very important in the dataset
    def make_training_data(self):
        for label in self.Labels:
            print(label)
            # tqdm is a progress bar
            for f in tqdm(os.listdir(label)):
                
                try:
                    path = os.path.join(label, f)
                    # You don't have to convert to grayscale - is color a relevant feature?
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    
                    """ One hot encoding - what index is hot?
                    Cat = [0 1]
                    Dog [ 1 0] 
                    np.eye(x)[y]
                    """

                    self.training_data.append([np.array(img), np.eye(2)[self.Labels[label]]])
                    
                    if label == self.cats:
                        self.cat_count += 1
                    
                    elif label == self.dogs:
                        self.dog_count += 1
                
                except Exception as  e:
                    print((e))
                    pass

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Cats ",self.cat_count)
        print("Dogs", self.dog_count)


if REBUILD_DATA: 
    dogsvcats = DogsvCats()
    dogsvcats.make_training_data()

training_data = np.load("training_data.npy", allow_pickle=True)

#plt.imshow(training_data[10000][0], cmap = 'gray')
#plt.show()

# Take batches of data and classify it using a convolutional neural network to classify


class Net(nn.Module):
    
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        self.conv1 = nn.Conv2d(1, 32, 5) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(32, 64, 5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)

    

net = Net()

optimizer = optim.Adam(net.parameters(), lr = 0.001)
loss_function = nn.MSELoss()

# Making sure the data is torch acceptable and seperating into samples and labels
X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1  # lets reserve 10% of our data for validation
val_size = int(len(X)*VAL_PCT)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1)

batch_size = 100
epochs = 2

EPOCHS = 3

def train(net):
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    BATCH_SIZE = 100
    EPOCHS = 3
    for epoch in range(EPOCHS):
        for i in range(0, len(X_train), BATCH_SIZE): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
            #print(f"{i}:{i+BATCH_SIZE}")
            batch_X = X_train[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
            batch_y = y_train[i:i+BATCH_SIZE]

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            net.zero_grad()

            optimizer.zero_grad()   # zero the gradient buffers
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()    # Does the update

        print(f"Epoch: {epoch}. Loss: {loss}")

    torch.save(net, "/home/parvfect/Documents/machine_learning/models/catdog")

def test(net):
    correct = 0
    total = 0
    for i in tqdm(range(0, len(X_test), BATCH_SIZE)):

        batch_X = X_test[i:i+BATCH_SIZE].view(-1, 1, 50, 50).to(device)
        batch_y = y_test[i:i+BATCH_SIZE].to(device)
        batch_out = net(batch_X)

        out_maxes = [torch.argmax(i) for i in batch_out]
        target_maxes = [torch.argmax(i) for i in batch_y]
        for i,j in zip(out_maxes, target_maxes):
            if i == j:
                correct += 1
            total += 1
    print("Accuracy: ", round(correct/total, 3))

if REBUILD_MODEL:
    train(net)
    model = net
else:
    model = torch.load("/home/parvfect/Documents/machine_learning/models/catdog")

test(model)
import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


REBUILD_DATA = True # set to true to one once, then back to false unless you want to change something in your training data.
REBUILD_MODEL = True

class beesVSants():
    IMG_SIZE = 50
    ants = "datasets/antbee/train/ants"
    bees = "datasets/antbee/train/bees"
    ants_test = "datasets/antbee/val/ants"
    bees_test = "datasets/antbee/val/bees"
    TESTING = "datasets/antbee/val"
    LABELS = {ants: 0, bees: 1}
    labels_testing = {ants_test:0, bees_test : 1}
    training_data = []
    testing_data = []

    catcount = 0
    dogcount = 0

    catcount_test = 0
    dogcount_test = 0 

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])  # do something like print(np.eye(2)[1]), just makes one_hot 
                        #print(np.eye(2)[self.LABELS[label]])

                        if label == self.ants:
                            self.catcount += 1
                        elif label == self.bees:
                            self.dogcount += 1

                    except Exception as e:
                        pass
                        #print(label, f, str(e))

        np.random.shuffle(self.training_data)
        np.save("ABtraining_data.npy", self.training_data)
        print('ants:',beesvants.catcount)
        print('bees:',beesvants.dogcount)

    def make_testing_data(self):
        for label in self.labels_testing:
            print(label)
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])  # do something like print(np.eye(2)[1]), just makes one_hot 
                        #print(np.eye(2)[self.LABELS[label]])

                        if label == self.ants_test:
                            self.catcount_test += 1
                        elif label == self.bees_test:
                            self.dogcount_test += 1

                    except Exception as e:
                        pass
                        #print(label, f, str(e))

        np.random.shuffle(self.testing_data)
        np.save("ABtesting_data.npy", self.testing_data)
        print('ants:',beesvants.catcount_test)
        print('bees:',beesvants.dogcount_test)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)



if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")




net = Net().to(device)

if REBUILD_DATA:
    beesvants = beesVSants()
    beesvants.make_training_data()
    beesvants.make_testing_data()

training_data = np.load("ABtraining_data.npy", allow_pickle=True)
print(len(training_data))

testing_data = np.load("ABtesting_data.npy", allow_pickle = True)
print(len(testing_data))

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

X_test = torch.Tensor([i[0] for i in testing_data]).view(-1, 50, 50)
X_test = X/255.0
y_test = torch.Tensor([i[1] for i in testing_data])

train_X = X

train_y = y

test_X = X_test
test_y = y_test

print(len(train_X))
print(len(test_X))


def train(net):
    BATCH_SIZE = 100
    EPOCHS = 20
    for epoch in range(EPOCHS):
        batch_X = train_X.view(-1,1,50,50)
        batch_y = train_y

        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        net.zero_grad()
        outputs = net(batch_X)
        
        matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(outputs, batch_y)]
        in_sample_acc = matches.count(True)/len(matches)

        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

        print(loss)
        print("In sample accuracy : ", round(in_sample_acc,2))

    torch.save(net, "/home/parvfect/Documents/AI/models/antbee")


def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_X[i].view(-1, 1, 50, 50).to(device))[0]

            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total += 1
    print("Accuracy:", round(correct/total,3))


if REBUILD_MODEL:
    train(net)
    model = net
else:
    model = torch.load("/home/parvfect/Documents/AI/models/catdog")
    model.trainable = True
    train(model)

test(model)


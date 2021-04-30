
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential,utils
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf

train = pd.read_csv("datasets/ecg_classification/train.csv", header = None)
test = pd.read_csv("datasets/ecg_classification/test.csv", header = None)


data = pd.DataFrame(train)

# So it's got 188 columns so the last one is labels I suppose and it's not uniformly distributed - so we need to sample the motherfucker

# Apply undersampling
class_1 = data[data[187] == 1.0]
class_2 = data[data[187] == 2.0]
class_3 = data[data[187] == 3.0]
class_4 = data[data[187] == 4.0]
class_0 = data[data[187] == 0.0].sample(n = 8000)

data = pd.concat([class_1, class_2, class_3, class_4, class_0]).sample(frac = 1)

xtrain, xtest, ytrain, ytest = train_test_split(data.drop([187],axis = 1), data[187], test_size = 0.1)

xtrain = np.array(xtrain).reshape(xtrain.shape[0], xtrain.shape[1], 1)
xtest = np.array(xtest).reshape(xtest.shape[0], xtest.shape[1], 1)

#fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(25,2))
#ax.plot(data[data[187]==float(2)].sample(1).iloc[0,:186])
#plt.show()

""" Applying CNN """ 

neu = Sequential()

neu.add(Conv1D(filters=32, kernel_size=(3,), padding='same', activation='relu', input_shape = (xtrain.shape[1],1)))
neu.add(Conv1D(filters=64, kernel_size=(3,), padding='same', activation='relu')) 
neu.add(Conv1D(filters=128, kernel_size=(5,), padding='same', activation='relu'))    

neu.add(MaxPool1D(pool_size=(3,), strides=2, padding='same'))
neu.add(Dropout(0.5))

neu.add(Flatten())

neu.add(Dense(units = 512, activation='relu'))
neu.add(Dense(units = 1024, activation='relu'))
neu.add(Dense(units = 5, activation='softmax'))


# taken from stack overflow to prevent memory growth on GPU or something (black box for now)
neu.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
his = neu.fit(xtrain, ytrain, epochs = 10)


ypred_train = neu.predict(xtest)
y_lbl = [np.where(i == np.max(i))[0][0] for i in ypred_train]
mat = confusion_matrix(ytest, y_lbl)
fig, ax = plt.subplots(figsize=(7,7))
sns.heatmap(mat, annot = True)

print("Accuracy score of the predictions: {0}".format(accuracy_score(y_lbl, ytest)))

# Run on test data - secondly import new test data and try - thirdly try the other dataset as well

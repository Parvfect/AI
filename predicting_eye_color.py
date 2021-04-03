
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


label_encoder_x= LabelEncoder()  

data = pd.read_csv("/home/parv/Documents/machine_learning/Eye-color-dataset/iris.csv")

""" Checking if data has a null value """
null_columns = data.columns[data.isnull().any()]
data[null_columns].isnull().sum()

""" Checking for categorical variables - converting to numeric """

#print(data['variety'].unique())

data["variety"] = data['variety'].astype('category')
data["variety"] = data['variety'].cat.codes
#print(data["variety"].unique())


""" Splitting into features and labels """
features = data.drop('variety', axis = 1).copy()
labels = data['variety'].copy()

""" Scaling and normalsing the data """
scaler = StandardScaler()
features = scaler.fit_transform(features)

""" Splitting into training and test sets """
X_train,X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.1, random_state = 123, shuffle = True)

#print(X_train.shape)
#print(y_train.shape)



inputs = tf.keras.Input(shape = (X_train.shape[1]))
x = tf.keras.layers.Dense(256, activation = 'relu')(inputs)
x = tf.keras.layers.Dense(256, activation = 'relu')(x)
x = tf.keras.layers.Dense(256, activation = 'relu')(x)
x = tf.keras.layers.Dense(256, activation = 'relu')(x)
outputs = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer = 'Adam', 
    loss = 'binary_crossentropy',
    metrics = [
        'accuracy',
        tf.keras.metrics.AUC(name='auc')
        ]
)

batch_size = 12
epochs = 50


history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau()
    ]
)

plt.figure(figsize=(16, 10))

plt.plot(range(epochs), history.history['loss'], label="Training Loss")
plt.plot(range(epochs), history.history['val_loss'], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Time")
plt.legend()

plt.show()


model.evaluate(X_test, y_test)


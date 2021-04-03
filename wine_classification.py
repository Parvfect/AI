import numpy as np
import pandas as pd
from time import time
from IPython.display import display
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from ml_preprocessing import univarite_analysis, multivariate_analysis, correlation_matrix



data = pd.read_csv("winequality-white.csv", sep=";")


# Checking if any missing values
print(data.isnull().any())


print(data['quality'].unique())

"""
data['quality'].plot(kind = 'hist')
plt.show()

pd.plotting.scatter_matrix(data, alpha = 0.3,
figsize = (40,40), diagonal = 'kde')
plt.show()

correlation_matrix(data)
"""
features = data.drop(['quality'], axis = 1).copy()
labels = data['quality'].copy()
labels = labels/10
print(labels)
print(features)

n_features = len(features.columns)
print(n_features)


scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=102, shuffle = True)
n_features = 10

try:
    model = tf.keras.models.load_model("/home/parv/Documents/NeuralData/wine_classification_model")

except:

    inputs = tf.keras.Input(shape = X_train.shape[1])

    x = tf.keras.layers.Dense(256, activation = 'relu')(inputs)

    x = tf.keras.layers.Dense(256, activation = 'relu')(x)

    outputs = tf.keras.layers.Dense(1, activation='relu')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer = 'Adam', 
        loss = 'mean_squared_error',
        metrics = [
            'accuracy',
            tf.keras.metrics.AUC(name='auc')
            ]
    )

    batch_size = 40
    epochs = 4


    history = model.fit(
        X_train,
        y_train, 
        validation_split = 0.1, 
        batch_size = batch_size, 
        epochs = epochs, 
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


results = model.evaluate(X_test, y_test)
print(results)

model.save("/home/parv/Documents/NeuralData/wine_classification_model")

import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report


sns.set(style='darkgrid', color_codes=True)

eeg_df = pd.read_csv('input/EEG_data.csv')
info_df = pd.read_csv('input/demographic_info.csv')


info_df.rename(columns={'subject ID': 'SubjectID'}, inplace=True)

data = info_df.merge(eeg_df, on='SubjectID')
data = data.drop(['SubjectID', 'VideoID', 'predefinedlabel', ' ethnicity'], axis=1)

data.rename(columns={' age': 'Age', ' gender': 'Gender', 'user-definedlabeln': 'Label'}, inplace=True)

data['Label'] = data['Label'].astype(np.int)


data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'M' else 0)


# Checking for non numeric columns
print("Non numeric Columns = " ,len(data.select_dtypes('object').columns))


features = data.drop('Label', axis = 1).copy()
num_features = len(features.columns)
print("Number of features = ", num_features)


categorical_features = ["Gender", "Age"]
continuous_features = ["Attention", "Mediation", "Raw", "Delta", "Theta", "Alpha1","Alpha2","Beta1", "Beta2", "Gamma1", "Gamma2"]


def univarite_analysis(df, continuous):
    """ References - (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html) """
    
    # If the features are continuous
    if(continuous):
        
        # Box Plots for single analysis
        df.plot(kind='box', figsize=(15, 15), subplots=True, layout=(3, 4))
        plt.show()

        # Histograms for single analysis
        df.plot(kind='hist', bins=25, figsize=(15, 12), subplots=True, layout=(3, 4))
        plt.show()

        #Line Plots for single analysis
        df.plot(figsize=(15, 12), subplots=True, layout=(3, 4))
        plt.show()

    # If the features are categorical
    else:

        # Pie plot
        plt.figure(figsize=(20, 5))
        for feature in df:
            plt.subplot(1, 5, categorical_features.index(feature) + 1)
            features[feature].value_counts().plot(kind='pie')
        plt.show()

def label_analysis(data):
    plt.figure(figsize=(8, 8))
    data['Label'].value_counts().plot(kind='pie', autopct='%.1f%%')
    plt.show()

"""
univarite_analysis(features[continuous_features], True)
univarite_analysis(features[categorical_features], False)
label_analysis(data)
"""

def multivariate_analysis(df):
    plt.figure(figsize=(20, 20))
    sns.pairplot(df)
    plt.show()

def correlation_matrix(df):
    """ Outputs the correlation of each factor to each other, making it easier to 
    see what to keep and what to drop"""
    corr = df.corr()

    plt.figure(figsize=(18, 15))
    sns.heatmap(corr, annot=True, vmin=-1.0, cmap='mako')
    plt.show()

# multivariate_analysis(features[continuous_features])
# correlation_matrix(features[continuous_features])

""" Splitting/Scaling """

"""
Reference 
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
"""

y = data['Label'].copy()
X = data.drop('Label', axis = 1).copy()

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Make sure that the random state is a unique seed everytime
X_train,X_test,y_train, y_test  = train_test_split(X, y, train_size = 0.8, random_state = 123)

""" Training """

print(X_train.shape)
inputs = tf.keras.Input(shape = (X_train.shape[1]))
x = tf.keras.layers.Dense(256, activation = 'relu')(inputs)
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

batch_size = 32
epochs = 5


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


import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf

import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report



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

def feature_importance():
    importance=ada_classifier.feature_importances_

    std = np.std([tree.feature_importances_ for tree in ada_classifier.estimators_],
                axis=0)
    indices = np.argsort(importance)

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.barh(range(X.shape[1]), importance[indices],
        color="b",  align="center")

    plt.yticks(range(X.shape[1]), colum_names)
    plt.ylim([0, X.shape[1]])
    plt.show()
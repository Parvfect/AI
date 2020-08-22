import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

HOUSING_PATH = os.path.join("datasets","housing")

def load_housing_data(housing_path=HOUSING_PATH):
    """Extract the data from the csv file"""
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def data_visual():
    """Visualise the matrice of data"""
    housing.hist(bins = 50, figsize = (20,15))
    plt.show()                          

def create_train_test(data, test_ratio = 0.2):
    """Split the data into train and test set depending on the ratio"""
    return train_test_split(data, test_size = test_ratio, random_state = 42)

housing = load_housing_data()
train_set, test_set = create_train_test(housing)
housing["income_cat"] = pd.cut(housing["median_income"], bins = [0.,1.5,3.0,4.5,6.,np.inf], labels = [1,2,3,4,5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis =1 , inplace = True)


housing = strat_train_set.copy()
housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.4, 
    s = housing["population"]/100, label = 'population', figsize = (10,7), 
    c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = "True",)
#plt.legend()
#plt.show()

#Finding correlation between the factors
housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['population_per_household'] = housing['population']/housing['households']
corr_matrix = housing.corr()

#Preparing the data for the algorithm
housing = strat_train_set.drop('median_house_value', axis = 1)
housing_labels = strat_train_set["median_house_value"].copy()

#Filling the missing value of the total bedrooms with the median value
median = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median, inplace = True)

#Turning text and categorical data to numbers
ordinal_encoder = OrdinalEncoder()
housing_cat = housing[['ocean_proximity']]
housing_cat_enc = ordinal_encoder.fit_transform(housing_cat)

#Converting to one_hot
one_hot = OneHotEncoder()
housing_cat_1hot = one_hot.fit_transform(housing_cat)

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('std_scaler', StandardScaler()),
])

housing_num = housing.drop('ocean_proximity', axis = 1)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
housing_prepared = full_pipeline.fit_transform(housing)

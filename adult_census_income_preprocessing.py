
# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reading Dataset

dataset = pd.read_csv("adult.csv")

dataset

# Exploratory Data Analysis (EDA)

dataset.head()

dataset.info()

dataset.describe()

plt.figure(figsize=(5,5))
sns.heatmap(data=dataset.select_dtypes(int,float).corr(),annot=True,fmt='.2f')
plt.show()

plt.figure(figsize=(20,7))
sns.barplot(x=dataset['education'],y=dataset['education.num'],data=dataset,order=dataset.sort_values('education.num').education)
plt.show()

zeroes = np.where(dataset['capital.gain'] == 0)[0]
print(len(zeroes))
zeroes = np.where(dataset['capital.gain'] != 0)[0]
print(len(zeroes))

# Preparing Features and Lable

X = dataset.drop(['income','capital.gain','capital.loss'],axis=1)
y = dataset['income']

X

y

# Split Data Into Train And Test Set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

X_train

X_test

y_train

y_test

# Data Cleaning

dataset

dataset = dataset.replace('?',np.nan)

X

dataset.isnull().sum()

dataset[dataset.duplicated()]

dataset.drop_duplicates(inplace=True)

int(dataset.duplicated().sum())



from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

categorical_columns = X.select_dtypes(object).columns

ohe_cat_col = categorical_columns[0:7]
frequency_cat_col = [categorical_columns[-1]]

ohe_pipline = Pipeline([
    ('imputer',SimpleImputer(missing_values=np.nan,strategy='most_frequent')),
    ('ohe',OneHotEncoder(handle_unknown='ignore',sparse_output=False)),
])

frequency_pipline = Pipeline([
    ('imputer',SimpleImputer(missing_values=np.nan,strategy='most_frequent')),
    ('freq',ce.CountEncoder(normalize=True)),
])

ct = ColumnTransformer([
    ('onehot',ohe_pipline,ohe_cat_col),
    ('frequency',frequency_pipline,frequency_cat_col),
],remainder='passthrough')

X_train_new = ct.fit_transform(X_train)
X_test_new = ct.transform(X_test)

print(X_train_new[1])

X_train.head(2)

print(X_test_new[1])

X_test.head(2)


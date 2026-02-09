# Major Imports
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

#########################################
# Set Up
#########################################

filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'
df = pd.read_csv(filepath, header=None)

print(df.head(10))
headers = ["age", "gender", "bmi", "no_of_children", "smoker", "region", "charges"]
df.columns = headers

df.replace('?', np.nan, inplace=True)

#########################################
# Data Wrangling
#########################################
print(df.info())

# Handle Missing Data
# Smoker is categorical, replace with most common entry
is_smoker = df['smoker'].value_counts().idxmax()
df['smoker'] = df['smoker'].fillna(is_smoker)

# Age is continuous, replace with mean
mean_age = df['age'].astype('float').mean(axis=0)
df['age'] = df['age'].fillna(mean_age)

# Update Values
df[['age','smoker']] = df[['age','smoker']].astype('int')

# Check
print(df.info())

# Charges column has values more than 2 decimal places long.
# Update to 2 decimal places.
df[['charges']] = np.round(df[['charges']], 2)
print(df.head())

#########################################
# Exploratory Data Analysis (EDA)
#########################################

# Regression plot for charges with respect to bmi.
sns.regplot(x="bmi", y="charges", data=df, line_kws={"color" : "red"})
plt.ylim(0,)
plt.show()

# Box plot for charges with respect to smoker.
sns.boxplot(x="smoker", y="charges",data=df)
plt.show()

# Correlation Matrix for dataset.
print(df.corr())

#########################################
# Model Development
#########################################

# SLRM to predict charges using smoker. Print R^2 score.
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

lr = LinearRegression()
X = df[['smoker']]
Y = df[['charges']]

lr.fit(X,Y)
print(lr.score(X, Y))

# Use MLRM to predict charges using all database aspects. Print R^2 score.
# Can reuse variable Y and lr.
Z = df[["age", "gender", "bmi", "no_of_children", "smoker", "region"]]
lr.fit(Z, Y)
print(lr.score(Z, Y))

# Create training pipeline.
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
pipe = Pipeline(Input)
Z = Z.astype(float)
pipe.fit(Z, Y)
ypipe = pipe.predict(Z)
print(r2_score(Y, ypipe))

#########################################
# Model Refinement
#########################################

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
x_train, x_test, y_train, y_test = train_test_split(Z, Y, test_size=0.2, random_state=1)

RidgeModel = Ridge(alpha=0.1)
RidgeModel.fit(x_train, y_train)
yhat = RidgeModel.predict(x_test)
print(r2_score(y_test, yhat))

pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
RidgeModel.fit(x_train_pr, y_train)
y_hat = RidgeModel.predict(x_test_pr)
print(r2_score(y_test, y_hat))
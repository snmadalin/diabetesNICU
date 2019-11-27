# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 18:18:15 2019

@author: Madalin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
diabetes = pd.read_csv('diabetes.csv')
diabetes.head
#Visualise the data
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
x = diabetes[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = diabetes['Outcome']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
classifier = LogisticRegression(solver= 'lbfgs', multi_class='auto')
classifier.fit(X_train, y_train)



from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error

predictions = classifier.predict(X_test)

print(f'R^2 score: {r2_score(y_true=y_test, y_pred=predictions):.2f}')
print(f'MAE score: {mean_absolute_error(y_true=y_test, y_pred=predictions):.2f}')
print(f'EVS score: {explained_variance_score(y_true=y_test, y_pred=predictions):.2f}')
rp = sns.regplot(x=y_test, y=predictions)



# MAKE A PREDICTION
# # I have created a new entry(patient) which contains values that are not compatible with the diabetes 
input_variables = pd.DataFrame([[3, 23, 50, 5, 6, 3.6, 0.697, 45]])
pred = classifier.predict(input_variables)

# Now we predict the outcome as is follows:

# A value of "0" means that the  individual is unlikley to have diabetes
print ('How likley is to have diabetes?', pred)


import pickle 
      with open('nicu.pkl', 'wb') as file: pickle.dump(classifier, file)
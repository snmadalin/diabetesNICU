# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# 1) LOAD THE DATASET

# Import the data from the research project folder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
diabetes = pd.read_csv('diabetes.csv')

# 2) INSPECT THE DATASET

#Diabetes dataset columns
print(diabetes.columns)


# Diabetes dataset head
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

diabetes.head(n= 10)

# Diabetes dataset dimension of data
print("Diabetes dataset dimension of data: {}".format(diabetes.shape))

#Dependent variable of our dataset
print(diabetes.groupby('Outcome').size())
#Outcome plot
import seaborn as sns
sns.countplot(diabetes['Outcome'],label="Count")

#Diabates dataset information
diabetes.info()

# 3) DIABETES DATASET CORRELATION MATRIX
corr = diabetes.corr()
corr
%matplotlib inline
import seaborn as sns
sns.heatmap(corr, annot = True)

g = sns.heatmap(diabetes.corr(),cmap="Blues",annot=False)

corr = diabetes.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);



#VISUALIZE OUR DATASET
import matplotlib.pyplot as plt
diabetes.hist(bins=50, figsize=(20, 15))
plt.show()


# 4) DATA CLEANING AND TRANSFORMATION

# Calculate the median value for BMI
median_bmi = diabetes['BMI'].median()
# Substitute it in the BMI column of the
# dataset where values are 0
diabetes['BMI'] = diabetes['BMI'].replace(
    to_replace=0, value=median_bmi)


# Calculate the median value for BloodPressure
median_bloodpressure = diabetes['BloodPressure'].median()
# Substitute it in the BloodP column of the
# dataset where values are 0
diabetes['BloodPressure'] = diabetes['BloodPressure'].replace(
    to_replace=0, value=median_bloodpressure)



# Calculate the median value for Glucose
median_glucose = diabetes['Glucose'].median()
# Substitute it in the Glucose column of the
# dataset where values are 0
diabetes['Glucose'] = diabetes['Glucose'].replace(
    to_replace=0, value=median_glucose)



# Calculate the median value for SkinThickness
median_skinthickness = diabetes['SkinThickness'].median()
# Substitute it in the SkinThick column of the
# dataset where values are 0
diabetes['SkinThickness'] = diabetes['SkinThickness'].replace(
    to_replace=0, value=median_skinthickness)



# Calculate the median value for Insulin
median_insulin = diabetes['Insulin'].median()
# Substitute it in the Insulin column of the
# dataset where values are 0
diabetes['Insulin'] = diabetes['Insulin'].replace(
    to_replace=0, value=median_insulin)


#VISUALIZE THE DATASET
import matplotlib.pyplot as plt
diabetes.hist(bins=50, figsize=(20, 15))
plt.show()


# 5) SPLITTING THE DATASET

# Split the training dataset in 75% / 25%
from sklearn.model_selection import train_test_split
x = data[['temperature', 'humidity', 'windspeed']]
y = data['count']

train_set_X, test_set_Y = train_test_split(diabetes, test_size=0.25, random_state=42)


# Separate labels from the rest of the dataset
train_set_labels_X = train_set_X["Outcome"].copy()
train_set_X = train_set_X.drop("Outcome", axis=1)

test_set_labels_Y = test_set_Y["Outcome"].copy()
test_set_Y = test_set_Y.drop("Outcome", axis=1)

# 6) FEATURE SCALING

# Apply a scaler
from sklearn.preprocessing import MinMaxScaler as Scaler

scaler = Scaler()
scaler.fit(train_set_X)
train_set_scaled_X = scaler.transform(train_set_X)
test_set_scaled_Y = scaler.transform(test_set_Y)

# 7) SCALED VALUES
df = pd.DataFrame(data=train_set_scaled_X)
df.head()


# 8) COMPARING DECISION TREE VS LOGISTIC REGRESSION VS SVC VS NAIVE BAYES

# IMPORT ALL THE ALGORITHMS THAT WE NEED AND WANT TO APPLY
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor



# Import the sklearn utility to compare algorithms
from sklearn import model_selection

# Prepare an array which contains all the algorithms that will be applied
models = []
models.append(('LR', LogisticRegression(solver = 'lbfgs', multi_class= 'auto')))
models.append(('NB', GaussianNB()))
models.append(('SVC', SVC(gamma = 'scale', )))
models.append(('DTR', DecisionTreeRegressor()))



# Prepare the configuration to run the test
seed = 4
results = []
names = []
X = train_set_scaled_X
Y = train_set_labels_X



# Every SINGLE  algorithm is tested and THE results are collected and THEN printed
for name, model in models:
    kfold = model_selection.KFold(
        n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(
        model, X, Y, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (
        name, cv_results.mean(), cv_results.std())
    print(msg)



# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Logistic Regression Classification Report
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
LR_model = LogisticRegression(solver ='lbfgs',multi_class='auto')
LR_model.fit(train_set_scaled_X,train_set_labels_X)
LR_prediction = LR_model.predict(test_set_scaled_Y)
print( 'Logistic Regression Classification Report:\n', classification_report(test_set_labels_Y,LR_prediction))


print '\nClasification report:\n', classification_report(y_test, svm_1_prediction)
print ('\nConfussion matrix:\n',confusion_matrix(test_set_labels_Y, pred_y)


# Grid search cross validation
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

new_parameters = [{'penalty': ['l2'],
       'C': np.logspace(1,15,75)}]

LR_model=LogisticRegression(solver = 'lbfgs', multi_class= 'auto')
LR_cv=GridSearchCV(LR_model,new_parameters,cv=10, verbose = 5, n_jobs = -1)
LR_cv.fit(train_set_scaled_X, train_set_labels_X)
pred_y = LR_cv.predict(test_set_scaled_Y)

print("New parameters :(best parameters) ",LR_cv.best_params_)
print("Accuracy : "+"{:.2%}".format(LR_cv.best_score_))
print("Confusion Matrix:")
print("{}".format(confusion_matrix(test_set_labels_Y, pred_y)))


#Apply the parameters to the model and train it


# Create an instance of the algorithm using parameters
# from best_estimator_ property
#LR_estimator = LR_cv.best_estimator_

# Use the whole dataset to train the model
#X = np.append(train_set_scaled_X, test_set_scaled_Y, axis=0)
#Y = np.append(train_set_labels_X, test_set_labels_Y, axis=0)

# Train the model
#LR_estimator.fit(X, Y)



# MAKE A PREDICTION
# # I have created a new entry(patient) which contains values that are not compatible with the diabetes 
input_variables = pd.DataFrame([[3, 23, 50, 95, 6, 23.6, 0.697, 45]])
#Scale those values like we did with the previous ones
input_variables_scaled = scaler.transform(input_variables)

# Now we predict the outcome as is follows:
prediction = LR_cv.predict(input_variables_scaled)

# A value of "0" means that the  individual is unlikley to have diabetes
print ('How likley is to have diabetes?', prediction)




import pickle
     with open('diabetes.pkl', 'wb') as file: pickle.dump(LR_cv, file)
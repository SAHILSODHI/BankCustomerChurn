#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sahilsodhi
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings(action="ignore")

np.random.seed(42)

df = pd.read_csv('BankCustomerDataset.csv')
df.shape
# Check for null values in the dataset
df.isnull().sum()
# Get unique count for each variable
df.nunique()
# RowNumber, CustomerId and Surname will be irrelevant for the predictions.
df = df.drop(["RowNumber","CustomerId","Surname"],axis = 1)
df.head()
df.dtypes

#Exploratory Data Analysis
#Calculate number of customers who exited(0) the bank and the ones who were 
#retained(1) in percentage.
df['Exited'].value_counts(normalize=True) * 100

# We first review the 'Status' relation with categorical variables
fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.countplot(x='Geography', hue = 'Exited',data = df, ax=axarr[0][0])
sns.countplot(x='Gender', hue = 'Exited',data = df, ax=axarr[0][1])
sns.countplot(x='HasCrCard', hue = 'Exited',data = df, ax=axarr[1][0])
sns.countplot(x='IsActiveMember', hue = 'Exited',data = df, ax=axarr[1][1])

# Relations based on the continuous data attributes
fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
sns.boxplot(y='CreditScore',x = 'Exited', hue = 'Exited',data = df, ax=axarr[0][0])
sns.boxplot(y='Age',x = 'Exited', hue = 'Exited',data = df , ax=axarr[0][1])
sns.boxplot(y='Tenure',x = 'Exited', hue = 'Exited',data = df, ax=axarr[1][0])
sns.boxplot(y='Balance',x = 'Exited', hue = 'Exited',data = df, ax=axarr[1][1])
sns.boxplot(y='NumOfProducts',x = 'Exited', hue = 'Exited',data = df, ax=axarr[2][0])
sns.boxplot(y='EstimatedSalary',x = 'Exited', hue = 'Exited',data = df, ax=axarr[2][1])

train_df = df

print(len(train_df))

#Feature Engineering
train_df['BalanceSalaryRatio'] = train_df.Balance/train_df.EstimatedSalary
train_df['TenureByAge'] = train_df.Tenure/(train_df.Age)
train_df['CreditScoreGivenAge'] = train_df.CreditScore/(train_df.Age)


#find correlation of independent variables with the target variable.
correlation = train_df[train_df.columns[:]].corr()['Exited'][:].sort_values(ascending = True)
# View categorical columns
print(train_df.select_dtypes(include=['object']).copy().columns.values)
print(train_df.select_dtypes(include=['int']).copy().columns.values)

categorical_features = train_df.select_dtypes(include=['object']).copy().columns
for column in categorical_features:
        onehot_train_df = pd.get_dummies(train_df[column], prefix=column)
        train_df = train_df.drop(column, 1)
        train_df = train_df.join(onehot_train_df)
train_df = train_df.drop(["Gender_Female","Geography_France"],axis=1)

X = train_df.drop('Exited',axis=1)
y = train_df['Exited']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.decomposition import PCA
pca = PCA(n_components = 7)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Fit models
from sklearn.linear_model import LogisticRegression
logistic_classifier = LogisticRegression(C = 0.5, penalty = 'l2')
logistic_classifier.fit(X_train, y_train)

parameters = [{'C': [0.1,0.5,1,10,50,100], 'penalty':['l2']}]
grid_search = GridSearchCV(LogisticRegression(solver = 'liblinear'),parameters, cv=5, refit=True, verbose=0)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

y_pred = logistic_classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred)
print (((cm_lr[0][0]+cm_lr[1][1])*100)/(cm_lr[0][0]+cm_lr[1][1]+cm_lr[0][1]+cm_lr[1][0]), '% of testing data was classified correctly by Logistic Regression')

from sklearn.svm import SVC
SVC_classifier = SVC(C = 0.5, gamma = 0.1 , kernel = 'poly', probability = True)
SVC_classifier.fit(X_train, y_train)

y_pred = SVC_classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
cm_svc = confusion_matrix(y_test, y_pred)
print (((cm_svc[0][0]+cm_svc[1][1])*100)/(cm_svc[0][0]+cm_svc[1][1]+cm_svc[0][1]+cm_svc[1][0]), '% of testing data was classified correctly by Support Vector Machine')

from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)
y_pred = (y_pred > 0.5)
cm_rf = confusion_matrix(y_test, y_pred)
print (((cm_rf[0][0]+cm_rf[1][1])*100)/(cm_rf[0][0]+cm_rf[1][1]+cm_rf[0][1]+cm_rf[1][0]), '% of testing data was classified correctly by Random Forest')

#XGBoost
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier(gamma = 0.01,
                               learning_rate = 0.1,
                               max_depth = 7,
                               min_child_weight = 5,
                               n_estimators=20)

xgb_classifier.fit(X_train, y_train)
xg_pred = xgb_classifier.predict(X_test)
xg_pred = (xg_pred > 0.5)
cm_xgb = confusion_matrix(y_test, xg_pred)
print (((cm_xgb[0][0]+cm_xgb[1][1])*100)/(cm_xgb[0][0]+cm_xgb[1][1]+cm_xgb[0][1]+cm_xgb[1][0]), '% of testing data was classified correctly by Extreme Gradient Boosting')

# to initialise our ANN
from keras.models import Sequential
# to build layers of our ANN.  
from keras.layers import Dense

# Initialising the ANN, to define our ANN as a sequence of layers.
ann_classifier = Sequential()

# Adding the input layer and the first hidden layer
#init is to initialise the weights close to zero like stated in the steps.
# here the 5 nodes which is the first layer is expecting the input of 7 nodes 
#as input.
ann_classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu', input_dim = 7))

# Adding the second hidden layer, add it anyway.
ann_classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu'))

# Adding the output layer, we need probabilities to know if a customer will 
# leave the bank. In case of multiple classification, change the output_dim to 
# 3 if we have 3 classifications and change activation to softmax.
ann_classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN. Optimizer is how to optimise our weights. We use Shotastic
# Gradient Descent algorithm one of which is adam.
# loss is required in Schotastic Gradient Descent to optimise the weights. 
# The loss function is the logarithmic loss. If multiple categories then 
# categorical_crossentropy.
# weights will improve based on accuracy, where metrics is expecting a list.
ann_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set.Epoch is the number of times we are 
# training our ANN on the whole training set.
ann_classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 1)

# Making the Confusion Matrix
cm_ann = confusion_matrix(y_test, y_pred)
print (((cm_ann[0][0]+cm_ann[1][1])*100)/(cm_ann[0][0]+cm_ann[1][1]+cm_ann[0][1]+cm_ann[1][0]), '% of testing data was classified correctly by Artificial Neural Network')

from sklearn import metrics
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

# Add the models to the list that you want to view on the ROC plot
models = [
{
    'label': 'Logistic Regression',
    'model': logistic_classifier,
},
{
    'label': 'Support Vector Machine',
    'model': SVC_classifier,
},
{   
    'label': 'Random Forests',
    'model': rf_classifier,
},
{   
    'label': 'Extreme Gradient Boosting',
    'model': xgb_classifier,
},
{   
    'label': 'Artificial Neural Network',
    'model': ann_classifier,
}
]

# Below for loop iterates through models list
for m in models:
    model = m['model'] # select the model
    model.fit(X_train, y_train) # train the model
    y_pred=model.predict(X_test) # predict the test data
    if m['model'] == ann_classifier:
        # Compute False postive rate, and True positive rate
        fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict(X_test).ravel())
    else:
        # Compute False postive rate, and True positive rate
        fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:,1])
    # Calculate Area under the curve to display on the plot
    auc = metrics.roc_auc_score(y_test,model.predict(X_test))
    # Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc))
# Custom settings for the plot 
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show() 

from sklearn import model_selection
print('5-fold cross validation:\n')

labels = ['Logistic Regression', 'Random Forest']

for clf, label in zip([logistic_classifier, rf_classifier], labels):

    scores = model_selection.cross_val_score(clf, X, y, 
                                              cv=5, 
                                              scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))

from sklearn.ensemble import VotingClassifier
voting_clf_hard = VotingClassifier(estimators = [(labels[0], logistic_classifier),
                                                 (labels[1], rf_classifier)],
                                   voting = 'hard')
voting_clf_soft = VotingClassifier(estimators = [(labels[0], logistic_classifier),
                                                 (labels[1], rf_classifier)],
                                   voting = 'soft')

labels_new = ['Logistic Regression', 'Random Forest',
              'Voting_Classifier_Hard', 'Voting_Classifier_Soft']

for (clf, label) in zip([logistic_classifier, rf_classifier, voting_clf_hard,
                        voting_clf_soft], labels_new):
    scores = model_selection.cross_val_score(clf, X, y, cv=5,
            scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))
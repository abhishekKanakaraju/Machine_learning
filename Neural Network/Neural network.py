# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 09:09:46 2021

@author: abhis
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

df1 = pd.read_excel(r'C:\Users\abhis\Downloads\BondRating.xls','Validation data')
df = pd.read_excel(r'C:\Users\abhis\Downloads\BondRating.xls','Training data')

class_1 = ['AAA','AA','A','BAA','BA','B','C']

print(class_1)
header_row = 1

df.columns = df.iloc[header_row]
df = df.iloc[2:]

df= df.drop(['OBS','RATING'], axis=1)


df1.columns = df1.iloc[header_row]
df1 = df1.iloc[2:]
df1= df1.drop(['OBS','RATING'], axis=1)


train_X = df.drop('CODERTG', axis=1)
valid_X = df1.drop('CODERTG', axis=1)
train_y = df['CODERTG']
valid_y = df1['CODERTG']

valid_y=valid_y.astype('int') 
train_y=train_y.astype('int') 

x=[]

from sklearn import metrics
for i in range(1,17):
    clf = MLPClassifier(hidden_layer_sizes=(i), activation='logistic', solver='lbfgs',random_state=1)
    clf.fit(train_X, train_y.values)
    y_pred = clf.predict(valid_X)
    x.append(metrics.accuracy_score(valid_y, y_pred)*100)
    
sr = pd.Series(x).to_frame()
sr.columns = ['Accuracy']
print(sr)

# training performance (use idxmax to revert the one-hotencoding)
train_report = classification_report(train_y, clf.predict(train_X))

# validation performance
test_report =classification_report(valid_y, clf.predict(valid_X))
test_Con_report=confusion_matrix(valid_y, clf.predict(valid_X))

print(test_Con_report)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(train_X, train_y)

y_pred = classifier.predict(valid_X)

from sklearn import tree
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(classifier, 
                    class_names = class_1,
                    filled=True)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(valid_y, y_pred)

print(cm)
print("Decision treee Accuracy:",metrics.accuracy_score(valid_y, y_pred)*100, " %")










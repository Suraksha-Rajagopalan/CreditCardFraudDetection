#Importing the necessary modules

import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import auc, classification_report
#Loading the dataset
data = pd.read_csv(r"C:\Users\araku\Downloads\ML\Project\archive\creditcard.csv", sep=",", header=0, engine='python')
data.head()
#TO find the number of columns and rows of the data
data.shape
#Description of the data
data.describe()
#Information of the data
data.info()
#Printing the number of non-null values
np.sum(data.isnull())
#Visualizing the null values
sn.heatmap(data.isnull())
#Finding the number of fraud(class 1) and non fraud(class 0) classes
data["Class"].value_counts()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
s = sn.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=data,  palette="PRGn",showfliers=True)
s = sn.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=data, showfliers=False)
plt.suptitle('Distribution of amount for Not Fraud and Fraud transaction')
plt.show();
#Selecting the input and target variables
Y = 'Class'
X = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',\
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',\
       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',\
       'Amount']
df = data.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()
#Rechecking the distributions of the classes
print('Distribution of the Classes in the subsample dataset')
print(new_df['Class'].value_counts()/len(new_df))

data = new_df
data
#Splitting into train, test and validation data
train_df, test_df = train_test_split(data, test_size=0.2, random_state = 101, shuffle=True )
train_df, valid_df = train_test_split(train_df, test_size=0.3, random_state= 101, shuffle=True )
#Using Random Forest

clf = RandomForestClassifier(n_jobs=4, 
                             random_state=234,
                             criterion='gini',
                             n_estimators=500,
                             max_depth = 10,
                             verbose=False)
clf.fit(train_df[X], train_df[Y].values)
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

for i in range(5):  # n  number of tree decision tree
    print("\n\nTree:", i+1)
    tree = clf.estimators_[i]        #For each tree
    dot_data = export_graphviz(tree,
                               feature_names=train_df[X].columns,  
                               filled=True,  
                               max_depth=2, 
                               impurity=False, 
                               proportion=True)
    graph = graphviz.Source(dot_data)
    display(graph)

#Predicting the Y values for validation data
preds = clf.predict(valid_df[X])
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(valid_df[Y].values, preds)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

entire_data = clf.predict(data[X])
#Confusion matrix
sn.heatmap(cm(data[Y].values, entire_data), annot=True)
#Confusion matrix
sn.heatmap(cm(valid_df[Y].values, preds), annot=True)
#Classification Report
classification_report(valid_df[Y].values, preds)


#Using AdaBoost

clf = AdaBoostClassifier(random_state=234,
                         algorithm='SAMME.R',
                         learning_rate=0.8,
                        n_estimators=200)
clf.fit(train_df[X], train_df[Y].values)
#Predicting the Y values for validation data
preds_Ada = clf.predict(valid_df[X])
# ROC AUC score
roc_auc_score(valid_df[Y].values, preds_Ada)
#Confusion matrix

sn.heatmap(cm(valid_df[Y].values, preds_Ada), annot=True)
#Roc curve for AdaBoost

roc_curve(valid_df[Y].values, preds_Ada)
fpr, tpr, thresholds = roc_curve(valid_df[Y].values, preds_Ada)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
#Classification Report

classification_report(valid_df[Y].values, preds_Ada)


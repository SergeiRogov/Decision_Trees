import pandas as pd
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# loading wine dataset
data = load_wine()

# code for loading breast cancer dataset (a bigger one)
# data = load_breast_cancer()

dataset = pd.DataFrame(data=data['data'], columns=data['feature_names'])

# splitting dataset - 70% for training and 30% for testing
X = dataset.copy()
Y = data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# 2 parameters used to plot a graph (there was a for loop with 'i' for it)
# prun_param = 0.005 * i
# depth_param = i+1

# creating decision tree model
classifier = DecisionTreeClassifier()
classifier = classifier.fit(X_train, Y_train)

# testing a model with a test set
predict = classifier.predict(X_test)

# MEASURING ACCURACY
# 'Handmade' accuracy = correct/total

# WINE DATASET: accuracy = 0.907
# BREAST CANCER DATASET: accuracy = 0.947
correct_classifications = 0
for j in range(len(Y_test)):
    if Y_test[j] == predict[j]:
        correct_classifications += 1
accuracy = correct_classifications/len(Y_test)

# Printing out a report on accuracy
print(f"Accuracy = {accuracy}")
print(f"Scikit accuracy = {accuracy_score(Y_test, predict)}")  # the same with handmade
print("Confusion Matrix")
print(confusion_matrix(Y_test, predict, labels=[0, 1]))
print(f"Precision = {precision_score(Y_test, predict, average=None)}")
print(f"Recall = {recall_score(Y_test, predict, average=None)}")
print(classification_report(Y_test, predict))

# Determining which features are more important for our model
# Presenting them in a sorted order and making a bar chart for visual representation
feature_names = X.columns
feature_importance = pd.DataFrame(classifier.feature_importances_,
                                  index=feature_names).sort_values(0, ascending=False)
feature_importance.plot.bar(title='Importance')
print(feature_importance)
features = data.feature_names
classes = data.target_names
plt.figure('Decision tree', figsize=(15, 10))
plot_tree(classifier,
          feature_names=features,
          class_names=classes,
          rounded=True,  # Rounded node edges
          filled=True,  # Adds color according to class
          proportion=True)  # Displays the proportions of class samples instead of the whole number of samples
# Code shows 2 graphs:
# 1) Bar chart of features importance
# 2) Decision tree
plt.show()

# A piece of code to plot a graph "parameter-accuracy"
# Works only if a for loop going on, collecting values of accuracy with a small step of a parameter

# pruning_parameters.append(prun_param)
# accuracy_array.append(accuracy)
# data_param = {'Pruning parameter': pruning_parameters,
#               'Accuracy': accuracy_array}
# table = pd.DataFrame(data_param)
# table.plot(x='Pruning parameter', y='Accuracy', kind='line', style='.-')
# plt.show()

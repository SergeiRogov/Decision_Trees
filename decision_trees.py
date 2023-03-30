import pandas as pd
from sklearn.datasets import load_wine
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
dataset = pd.DataFrame(data=data['data'], columns=data['feature_names'])
# print(dataset)

# splitting dataset - 70% for training and 30% for testing
X = dataset.copy()
Y = data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

pruning_parameters = []
accuracy_array = []
for i in range(20):
    # creating decision tree model
    prun_param = 0.005 * i
    print(f'Number {i}, prun_param = {prun_param}')
    classifier = DecisionTreeClassifier(ccp_alpha=prun_param)
    classifier = classifier.fit(X_train, Y_train)

    # testing a model with a test set
    predict = classifier.predict(X_test)
    # print(classifier.predict_proba(X_test))

    # Measuring ACCURACY
    # 'Handmade' accuracy = correct/total
    # Is equal to 0.85185
    correct_classifications = 0
    for j in range(len(Y_test)):
        if Y_test[j] == predict[j]:
            correct_classifications += 1
    accuracy = correct_classifications/len(Y_test)
    print(f"Accuracy = {accuracy}")
    print(f"Scikit accuracy = {accuracy_score(Y_test, predict)}")
    print("Confusion Matrix")
    print(confusion_matrix(Y_test, predict, labels=[0, 1]))
    print(f"Precision = {precision_score(Y_test, predict, average=None)}")
    print(f"Recall = {recall_score(Y_test, predict, average=None)}")
    print(classification_report(Y_test, predict))

    # feature_names = X.columns
    # feature_importance = pd.DataFrame(classifier.feature_importances_,
    #                                   index=feature_names).sort_values(0, ascending=False)
    # feature_importance.plot.bar(title='Importance')
    #
    # features = data.feature_names
    # classes = data.target_names
    # plt.figure('Decision tree', figsize=(15, 8))
    # plot_tree(classifier,
    #           feature_names=features,
    #           class_names=classes,
    #           rounded=True,  # Rounded node edges
    #           filled=True,  # Adds color according to class
    #           proportion=True)  # Displays the proportions of class samples instead of the whole number of samples
    # plt.show()
    pruning_parameters.append(prun_param)
    accuracy_array.append(accuracy)

data_param = {'Pruning parameter': pruning_parameters,
              'Accuracy': accuracy_array}
table = pd.DataFrame(data_param)
print(table)
table.plot(x='Pruning parameter', y='Accuracy', kind='line', style='.-')
plt.show()

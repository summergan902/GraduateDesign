import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import pandas as pd

# Importing the dataset
y = []
train_a = []
x = np.genfromtxt('train.csv',delimiter=',')
f = open("labels_1.dat","r")
for i in f:
	y.append(i)
y = np.array(y).astype(np.float)
y = y.astype(np.int)
y[y<=5] = 0 #低效价
y[y>5] = 1 #高效价
print(y)
x = np.array(x)
#scaler = StandardScaler()
#x_std = scaler.fit_transform(x)
#fitting NAIVE BAYES CLASSIFIER to the training set

stratified_folder = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
score_list = []
for train_index, test_index in stratified_folder.split(x, y):
    classifier = GaussianNB()
    x_train = x[train_index]
    y_train = y[train_index]
    x_test = x[test_index]
    y_test = y[test_index]
    classifier.fit(x_train, y_train)
    score = classifier.score(x_test, y_test)
    score_list.append(score)

print(len(score_list))
print("NB final--------->  ",np.mean(score_list))


# 下面是可视化的内容
# #visualizing the training set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_train, y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step=0.01),
#                      np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step=0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap= ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X2.max())
# plt.ylim(X2.min(), X1.max())
# for i,j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set==j, 0], X_set[y_set==j, 1],
#                 c= ListedColormap(('red', 'green'))(i), label= j)
#
# plt.title('Naive bayes (training set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.show()
#
#
# #visualizing the test set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_test, y_test
# X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step=0.01),
#                      np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step=0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap= ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X2.max())
# plt.ylim(X2.min(), X1.max())
# for i,j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set==j, 0], X_set[y_set==j, 1],
#                 c= ListedColormap(('red', 'green'))(i), label= j)
#
# plt.title('Naive Bayes (test set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.show()
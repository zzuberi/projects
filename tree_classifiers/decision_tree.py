import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import tree
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.1, random_state=0)


clf = tree.DecisionTreeClassifier(random_state=0)
clf = clf.fit(X_train, y_train)

y_prediction = clf.predict(X_test)
print(sum(np.equal(y_prediction, y_test)))

tree.plot_tree(clf)




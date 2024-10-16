from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
from sklearn import tree


data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

rf = RandomForestClassifier(n_estimators=3, criterion='gini', bootstrap=True, max_samples=0.5, random_state=42,
                            max_depth=3)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)



print(acc)
print(cm)

print(data.feature_names, rf.feature_importances_)



tree1 = rf.estimators_[0]
plt.figure(figsize=(10, 10))
tree.plot_tree(tree1, feature_names=data.feature_names, class_names=['1', '2', '3'])
plt.show()
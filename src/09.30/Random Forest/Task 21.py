from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd


def load_data():
    """
    Loads the breast cancer dataset and returns the features and target data.
    """
    data = load_breast_cancer()
    return data['data'], data['target'], data['feature_names'], data['target_names']


def train_decision_tree(X_train, y_train, criterion='gini'):
    """
    Trains a decision tree classifier using the specified criterion.

    Args:
        X_train (ndarray): The training features.
        y_train (ndarray): The training labels.
        criterion (str): The criterion for the decision tree ('gini' or 'entropy').

    Returns:
        DecisionTreeClassifier: The trained decision tree model.
    """
    clf = DecisionTreeClassifier(criterion=criterion, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_test, y_test):
    """
    Evaluates the trained model on the test data, returning accuracy and confusion matrix.

    Args:
        clf (DecisionTreeClassifier): The trained classifier.
        X_test (ndarray): The test features.
        y_test (ndarray): The test labels.

    Returns:
        dict: Dictionary containing accuracy, confusion matrix, and classification report.
    """
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report
    }


def visualize_tree(clf, feature_names):
    """
    Visualizes the decision tree using matplotlib.

    Args:
        clf (DecisionTreeClassifier): The trained classifier.
        feature_names (ndarray): The names of the features.
    """
    plt.figure(figsize=(20, 10))
    plot_tree(clf, filled=True, feature_names=feature_names, class_names=['benign', 'malignant'], rounded=True)
    plt.show()


def important_features(clf, feature_names):
    """
    Prints the most important features based on the decision tree.

    Args:
        clf (DecisionTreeClassifier): The trained classifier.
        feature_names (ndarray): The names of the features.
    """
    importance = pd.Series(clf.feature_importances_, index=feature_names)
    print(importance.sort_values(ascending=False))


def main():
    # Load and split the data
    X, y, feature_names, target_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train and evaluate using Gini criterion
    print("Gini Criterion:")
    clf_gini = train_decision_tree(X_train, y_train, criterion='gini')
    gini_results = evaluate_model(clf_gini, X_test, y_test)
    print(f"Accuracy: {gini_results['accuracy']}")
    print("Confusion Matrix:\n", gini_results['confusion_matrix'])
    print("Classification Report:\n", gini_results['classification_report'])

    # Train and evaluate using Entropy criterion
    print("\nEntropy Criterion:")
    clf_entropy = train_decision_tree(X_train, y_train, criterion='entropy')
    entropy_results = evaluate_model(clf_entropy, X_test, y_test)
    print(f"Accuracy: {entropy_results['accuracy']}")
    print("Confusion Matrix:\n", entropy_results['confusion_matrix'])
    print("Classification Report:\n", entropy_results['classification_report'])

    # Visualize the trees
    print("\nVisualizing Decision Tree (Gini):")
    visualize_tree(clf_gini, feature_names)

    print("\nVisualizing Decision Tree (Entropy):")
    visualize_tree(clf_entropy, feature_names)

    # Important features
    print("\nImportant Features (Gini):")
    important_features(clf_gini, feature_names)

    print("\nImportant Features (Entropy):")
    important_features(clf_entropy, feature_names)


if __name__ == "__main__":
    main()

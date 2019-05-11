from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import random
import numpy as np
import pandas as pd

def RandomForestTrees(adult_data, adult_labels):
    tree_count = 10
    bag_proportion = 0.6
    predictions = []

    train_data, test_data, train_labels, test_labels = train_test_split(adult_data, adult_labels, test_size=0.2,
                                                                        random_state=random.randint(1, 99))

    for i in range(tree_count):
        # bag = train_data.sample(frac=bag_proportion, replace=True, random_state=i)
        X_train, X_test = train_data, test_data
        y_train, y_test = train_labels, test_labels
        clf = DecisionTreeClassifier(random_state=1, min_samples_leaf=75)
        clf.fit(X_train, y_train)
        predictions.append(clf.predict_proba(X_test)[:, 1])

    combined = np.sum(predictions, axis=0) / 10
    rounded = np.round(combined)

    print(accuracy_score(rounded, y_test))
    print(roc_auc_score(rounded, y_test))
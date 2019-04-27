from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from Enchancers.MachineEnchancers import quantileTransformationUniformScaler_Custom
from sklearn.feature_selection import SelectKBest, chi2

import random
import numpy as np


def logReg(adult_data, adult_labels, feature_selection = 14):

    #Split Data
    train_data, test_data, train_labels, test_labels = train_test_split(adult_data, adult_labels, test_size=0.2,
                                                                        random_state=random.randint(1, 99))

    # Preprocess Data
    # This Scaler is used according to the results obtained in DataScalaionComparison for Logistical Regression
    train_data, test_data = quantileTransformationUniformScaler_Custom(train_data, test_data)

    # Feature Selection
    skb = SelectKBest(chi2, k=feature_selection)
    adult_data_selected = skb.fit_transform(adult_data, adult_labels)
    adult_data = adult_data_selected

    cols = skb.get_support(indices=True)
    print(cols)

    #Fit Machine
    logreg = LogisticRegression(C = 0.1, max_iter= 1000, penalty= "l1", solver="saga")
    logreg.fit(train_data, train_labels)

    # Best Parameters
    # Used Grid Search CV to find the best hyper-parameters to tune Logistical Regression machine.
    # Winning combination: {'C': 0.1, 'max_iter': 1000, 'penalty': 'l1', 'solver': 'saga'}
    # Gained around 0.8-1 % accuracy

    # grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1"], "solver": ["saga"], "max_iter": [100, 1000, 10000, 100000]}  # l1 lasso l2 ridge
    # logreg_cv = GridSearchCV(logreg, grid, cv=5)
    # logreg_cv.fit(train_data, train_labels)

    # print("tuned hpyerparameters :(best parameters) ", logreg_cv.best_params_)
    # print("accuracy :", logreg_cv.best_score_)
    # print(logreg_cv.cv_results_)

    #Predict
    logreg_prediction = logreg.predict(test_data)

    return np.count_nonzero(np.equal(test_labels, logreg_prediction)) * 100 / logreg_prediction.size
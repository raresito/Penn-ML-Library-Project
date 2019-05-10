from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from Enchancers.MachineEnchancers import quantileTransformationUniformScaler_Custom
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.model_selection import GridSearchCV

import random
import numpy as np


def sgdcalssifier(adult_data, adult_labels):

    #Split Data
    train_data, test_data, train_labels, test_labels = train_test_split(adult_data, adult_labels, test_size=0.2,
                                                                        random_state=random.randint(1, 99))

    # Preprocess Data
    # This Scaler is used according to the results obtained in DataScalaionComparison for Logistical Regression
    train_data, test_data = quantileTransformationUniformScaler_Custom(train_data, test_data)

    # Feature Selection
    # skb = SelectKBest(chi2, k=14)
    # adult_data_selected = skb.fit_transform(adult_data, adult_labels)
    # adult_data = adult_data_selected
    # rfe_selector =


#     cols = skb.get_support(indices=True)
#     print(cols)

    #Fit Machine
    sgdc = SGDClassifier(alpha=0.001, loss='log', max_iter=1000, penalty='l1')
    sgdc.fit(train_data, train_labels)

    # Best Parameters
    # Used Grid Search CV to find the best hyper-parameters to tune Logistical Regression machine.
    # Winning combination: {'C': 0.1, 'max_iter': 1000, 'penalty': 'l1', 'solver': 'saga'}
    # Gained around 0.8-1 % accuracy

    # Need more tests
    # grid = {"loss": ["hinge", "log", "modified_huber", "squared_hinge"],
    #         "penalty": ["none", "l2", "l1", "elasticnet"],
    #         "alpha": [0.0001, 0.001, 0.01, 0.1],
    #         "max_iter": [100, 1000, 10000, 100000]}

    # # Most complex test run.
    # grid = {"loss": ["hinge", "log"],
    #         "penalty": ["none", "l2", "l1", "elasticnet"],
    #         "alpha": [0.0001, 0.001],
    #         "max_iter": [100,1000]}
    #
    # # Result of complex test
    # # {'alpha': 0.001, 'loss': 'log', 'max_iter': 1000, 'penalty': 'l1'}
    #
    # sgdc_cv = GridSearchCV(sgdc, grid, cv=5)
    # sgdc_cv.fit(train_data, train_labels)
    #
    # print("tuned hpyerparameters :(best parameters) ", sgdc_cv.best_params_)
    # print("accuracy :", sgdc_cv.best_score_)
    # print(sgdc_cv.cv_results_)

    #Predict
    sgdc_prediction = sgdc.predict(test_data)

    return np.count_nonzero(np.equal(test_labels, sgdc_prediction)) * 100 / sgdc_prediction.size
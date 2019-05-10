from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from Enchancers.MachineEnchancers import quantileTransformationUniformScaler_Custom
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
import matplotlib
import ModelEvaluation

import random
import numpy as np
import pandas as pd
from IPython.display import display

def supportVectorMachine(adult_data, adult_labels):

    #Split Data
    train_data, test_data, train_labels, test_labels = train_test_split(adult_data, adult_labels, test_size=0.2,
                                                                        random_state=random.randint(1, 99))

    # Preprocess Data
    # This Scaler is used according to the results obtained in DataScalaionComparison for Logistical Regression
    train_data, test_data = quantileTransformationUniformScaler_Custom(train_data, test_data)

    # Feature Selection
    skb = SelectKBest(chi2, k=13)
    adult_data_selected = skb.fit_transform(adult_data, adult_labels)
    adult_data = adult_data_selected

    # cols = skb.get_support(indices=True)
    # print(cols)

    #Fit Machine
    LSVC = SVC(kernel='linear', random_state=0, tol=1e-5, C=0.1)
    LSVC.fit(train_data, train_labels)

    PolySVM = SVC(kernel='poly')
    PolySVM.fit(train_data, train_labels)

    RBFSVM = SVC(kernel='rbf', C = 1, tol = 1e-3)
    RBFSVM.fit(train_data, train_labels)

    SigmoidSVM = SVC(kernel='sigmoid')
    SigmoidSVM.fit(train_data, train_labels)


    #Predict
    LSVC_prediction = LSVC.predict(test_data)
    LSVC_model = m
    
    Poly_prediction = PolySVM.predict(test_data)
    RBFSVM_prediction = RBFSVM.predict(test_data)
    SigmoidSVM_prediction = SigmoidSVM.predict(test_data)

    ovl_svm = round(pd.DataFrame([RBFSVM, LSVC, PolySVM, SigmoidSVM],
                                 index=['SVM_rbf', 'SVM_linear', 'SVM_poly', 'SVM_sigmoid']), 4)
    display(ovl_svm)

    # return np.count_nonzero(np.equal(test_labels, sgdc_prediction)) * 100 / sgdc_prediction.size
    return 1



##

# Best Parameters
    # Used Grid Search CV to find the best hyper-parameters to tune Logistical Regression machine.
    # Winning combination: {'C': 0.1, 'loss': 'squared_hinge', 'penalty': 'l1'}
    # Gained around

    # Need more tests
    # grid = {"loss": ["hinge", "squared_hinge"],
            # "C": [0.001, 0.01, 0.1, 1, 10, 100]}

    # Most complex test run.
    # grid = {"penalty": ["l1", "l2"],
    #         "loss": ["hinge", "squared_hinge"],
    #         "C": [0.001, 0.01, 0.1, 1, 10, 100]
    #         }
    # # Result of complex test:
    # # accuracy : 0.8270928774345456
    #
    # LSVC_cv = GridSearchCV(LSVC, grid, cv=5)
    # LSVC_cv.fit(train_data, train_labels)
    # #
    # print("tuned hpyerparameters :(best parameters) ", LSVC_cv.best_params_)
    # print("accuracy :", LSVC_cv.best_score_)
    # print(LSVC_cv.cv_results_)
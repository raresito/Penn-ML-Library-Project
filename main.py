from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.metrics import classification_report
import numpy as np
import random
import matplotlib.pyplot as plt

# Choose data set.
adult_data, adult_labels = fetch_data('adult', return_X_y=True, local_cache_dir='./')
print(adult_data.shape, adult_labels.shape)


# Algorithms to be used
logreg = LogisticRegression( solver='lbfgs' )
gaussNB = GaussianNB()
sgd = SGDClassifier( loss="hinge", penalty="l2", max_iter=5 )
linear = LinearRegression()
rfc = RandomForestClassifier(n_estimators=200)

# Columns used to create predictions
feature_columns = ['age', 'workclass', 'education', 'education-num',
       'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

# Feature Selection
adult_data_selected = SelectKBest(chi2, k = 5).fit_transform(adult_data, adult_labels)
adult_data = adult_data_selected

# Meaning results
logreg_mean_results = []
gaussNB_mean_results = []
sgd_mean_results = []
linear_mean_results = []
rfc_mean_results = []

# for x in range(20):

#Split Data into test and training sets.
train_data, test_data, train_labels, test_labels = train_test_split(adult_data, adult_labels, test_size=0.2, random_state = random.randint(1,99))

# Applying Standard Scaling
print(train_data)
sc = StandardScaler()
train_data = sc.fit_transform(train_data)
test_data = sc.transform(test_data)
print(train_data)

# Training ML
# print("Started training...")
logreg.fit(train_data,train_labels)
gaussNB.fit(train_data,train_labels)
sgd.fit(train_data,train_labels)
linear.fit(train_data, train_labels)
rfc.fit(train_data, train_labels)

# Testing some predictions
logreg_prediction = logreg.predict(test_data)
gaussNB_prediction = gaussNB.predict(test_data)
sgd_prediction = sgd.predict(test_data)
linear_prediction = linear.predict(test_data)
rfc_prediction = rfc.predict(test_data)

# Print predictions
logreg_result = np.count_nonzero(np.equal(test_labels, logreg_prediction)) * 100 / logreg_prediction.size
logreg_mean_results.append(logreg_result)
# print("Logistic Regression: ", logreg_result)

gaussNB_result = np.count_nonzero(np.equal(test_labels, gaussNB_prediction)) * 100 / gaussNB_prediction.size
gaussNB_mean_results.append(gaussNB_result)
# print("Naive Gaussian Regression: ", gaussNB_result)

sgd_result = np.count_nonzero(np.equal(test_labels, sgd_prediction)) * 100 / sgd_prediction.size
sgd_mean_results.append(sgd_result)
# print("Naive Gaussian Regression: ", sgd_result)

linear_result = np.count_nonzero(np.equal(test_labels, linear_prediction)) * 100 / linear_prediction.size
linear_mean_results.append(linear_result)
# print("Naive Gaussian Regression: ", linear_result)

rfc_result = np.count_nonzero(np.equal(test_labels, rfc_prediction)) * 100 / rfc_prediction.size
rfc_mean_results.append(rfc_result)
# print("Naive Gaussian Regression: ", rfc_result)

print("Mean Logistic Regression: ", np.mean(logreg_mean_results))
print("Mean Naive Gaussian Regression: ", np.mean(gaussNB_mean_results))
print("Mean Stochastic Gradient Descent: ", np.mean(sgd_mean_results))
print("Mean Linear Regression: ", np.mean(linear_mean_results))
print("Mean Random Forrest Classifier: ", np.mean(rfc_mean_results))

print(classification_report(test_labels, rfc_prediction))
print(classification_report(test_labels, logreg_prediction))
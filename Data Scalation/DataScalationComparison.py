# Author:  Raghav RV <rvraghav93@gmail.com>
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Thomas Unterthiner
# License: BSD 3 clause

from __future__ import print_function

import numpy as np
from pmlb import fetch_data
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_california_housing


print(__doc__)

adult_data, adult_labels = fetch_data('adult', return_X_y=True, local_cache_dir='./')

#Split Data into test and training sets.
train_data, test_data, train_labels, test_labels = train_test_split(adult_data, adult_labels, test_size=0.2, random_state = random.randint(1,99))


# Take only 2 features to make visualization easier
# Feature of 0 has a long tail distribution.
# Feature 5 has a few but very large outliers.

X_train = train_data
# print(train_data.shape, X_train.shape)
X_test = test_data
y_train = train_labels
y_test = test_labels

logreg = LogisticRegression( solver='lbfgs' )

distributions = [
    ('Unscaled data', X_train),
    ('Data after standard scaling',
        StandardScaler()),
    ('Data after min-max scaling',
        MinMaxScaler()),
    ('Data after max-abs scaling',
        MaxAbsScaler()),
    ('Data after robust scaling',
        RobustScaler(quantile_range=(25, 75))),
    ('Data after power transformation (Yeo-Johnson)',
     PowerTransformer(method='yeo-johnson')),
    # ('Data after power transformation (Box-Cox)',
    #  PowerTransformer(method='box-cox')),
    ('Data after quantile transformation (gaussian pdf)',
        QuantileTransformer(output_distribution='normal')
        ),
    ('Data after quantile transformation (uniform pdf)',
        QuantileTransformer(output_distribution='uniform')
        ),
    ('Data after sample-wise L2 normalizing',
        Normalizer()),
]

for i in range (1, len(distributions)):
    scaler = distributions[i][1]
    train_data = scaler.fit_transform(X_train);
    test_data = scaler.transform(X_test)

    logreg.fit(train_data, train_labels)

    logreg_prediction = logreg.predict(test_data)

    logreg_result = np.count_nonzero(np.equal(test_labels, logreg_prediction)) * 100 / logreg_prediction.size
    print(distributions[i][0], ": ", logreg_result)

# logreg.fit(train_data,train_labels)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

def standardScaler_Custom(train_data, test_data):
    sc = StandardScaler()
    train_data = sc.fit_transform(train_data)
    test_data = sc.transform(test_data)
    return train_data, test_data

def quantileTransformationUniformScaler_Custom(train_data, test_data):
    qt = QuantileTransformer(output_distribution='uniform')
    train_data = qt.fit_transform(train_data)
    test_data = qt.transform(test_data)
    return train_data, test_data
from pmlb import fetch_data, classification_dataset_names
from Regressions.SGDClassifier_Solver import sgdcalssifier
from Regressions.SupportVectorMachine_Solver import supportVectorMachine
from Regressions.RandomForestTree import RandomForestTrees
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from Enchancers.DataSanitization import sanitize

# Choose data set.
# adult_data, adult_labels = fetch_data('adult', return_X_y=True, local_cache_dir='./')

feature_columns = ['age', 'workclass', 'education', 'education-num',
       'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

adult_data_frame = fetch_data('adult')


# print(adult_data_frame.isnull().sum())
adult_data_frame = sanitize(adult_data_frame)
adult_labels = adult_data_frame.pop('target').values
adult_data = adult_data_frame[feature_columns]


# print(logReg(adult_data,adult_labels, 14))
# print(sgdcalssifier(adult_data, adult_labels))
print(supportVectorMachine(adult_data, adult_labels))
# RandomForestTrees(adult_data, adult_labels)

# print(adult_data_frame.head())

##################################

# Top features that predict income are:
# 1. Capital-gain
# 2. Capital-loss
# 3. Fnlwgt

################################################
# Test fo find the best number of features that should be used
# best_feature_selection = []
# for x in range(13):
#     this_list = []
#     for testing in range(10):
#         this_list.append( logReg(adult_data,adult_labels, True, x+1) )
#     best_feature_selection.append(this_list)
#
# print(best_feature_selection)
# list_of_means = []
# for i in range(len(best_feature_selection)):
#     list_of_means.append(np.mean(best_feature_selection[i]))
# print(list_of_means)
#
# new_list_of_means = []
# for i in range (len(list_of_means)):
#     new_list_of_means.append( (i+1, list_of_means[i]) )
#
# new_list_of_means.sort(key=lambda tup: tup[1])
# print(new_list_of_means)


#############################################
# Benchmark for Comparing scaled and un-scaled data with Standard Scaler
# Result were in favour of scaling with an improvement of around 3%
# withScale = []
# withoutScale = []
# for x in range(10):
#     withScale.append( logReg(adult_data,adult_labels, True))
#     withoutScale.append( logReg(adult_data,adult_labels, False))
# print(np.mean(withScale), np.mean(withoutScale))
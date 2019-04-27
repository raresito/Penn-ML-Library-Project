print(type(adult_data))
adult_data_scaled = preprocessing.scale(adult_data)
adult_data = pd.DataFrame( data = adult_data_scaled,
                           columns = adult_data.columns,
                           index = adult_data.index)
print(type(adult_data))

#################################

# adult_labels = [adult_labels[i] for i
#        in ch2.get_support(indices=True)]

# adult_data = adult_data_selected


####################################################
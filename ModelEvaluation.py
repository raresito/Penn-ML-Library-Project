import pandas as pd

def model_eval(actual, pred):
    confusion = pd.crosstab(actual, pred, rownames=['Actual'], colnames=['Predicted'])
    TP = confusion.loc['>50K', '>50K']
    TN = confusion.loc['<=50K', '<=50K']
    FP = confusion.loc['<=50K', '>50K']
    FN = confusion.loc['>50K', '<=50K']

    accuracy = ((TP + TN)) / (TP + FN + FP + TN)
    precision = (TP) / (TP + FP)
    recall = (TP) / (TP + FN)
    f_measure = (2 * recall * precision) / (recall + precision)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    error_rate = 1 - accuracy

    out = {}
    out['accuracy'] = accuracy
    out['precision'] = precision
    out['recall'] = recall
    out['f_measure'] = f_measure
    out['sensitivity'] = sensitivity
    out['specificity'] = specificity
    out['error_rate'] = error_rate

    return out
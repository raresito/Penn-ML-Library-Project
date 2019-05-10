from scipy.stats import pointbiserialr, spearmanr
import pandas as pd

def sanitize(adult_data_frame):

    new_data  = adult_data_frame

    # Group Couples and Singles
    new_data['marital-status'] = new_data['marital-status'].replace([0, 3, 4, 5, 6], 0)
    new_data['marital-status'] = new_data['marital-status'].replace([1, 2], 1)

    # Vizualize the correlation

    param = []
    correlation = []
    abs_corr = []

    col_names = new_data.columns

    for c in col_names:
        # Check if binary or continuous
        if c != "target":
            if len(new_data[c].unique()) <= 2:
                corr = spearmanr(new_data['target'], new_data[c])[0]
            else:
                corr = pointbiserialr(new_data['target'], new_data[c])[0]
            param.append(c)
            correlation.append(corr)
            abs_corr.append(abs(corr))

    # Create dataframe for visualization
    param_df = pd.DataFrame({'correlation': correlation, 'parameter': param, 'abs_corr': abs_corr})

    # Sort by absolute correlation
    param_df = param_df.sort_values(by=['abs_corr'], ascending=False)

    # Set parameter name as index
    param_df = param_df.set_index('parameter')

    # print(param_df)

    return new_data
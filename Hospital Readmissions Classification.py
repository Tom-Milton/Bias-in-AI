import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def clean_dataset(df, missing_percent_threshold):
    """Cleans dataset for use with logistic regression"""

    original_columns = df.columns

    # Relabel y
    df['readmitted'].replace({'NO': 0, '>30': 0, '<30': 1}, inplace=True)
    print(df[df['readmitted'] == 1].shape[0], 'positive labels')
    print(df[df['readmitted'] == 0].shape[0], 'negative labels', '\n')

    # Remove cases that result in death or discharge to hospice to prevent bias
    df.drop(df[df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])].index, inplace=True)

    # Remove duplicate patient encounters to prevent bias
    df.drop_duplicates('patient_nbr', inplace=True)
    print('dataframe has shape', df.shape, 'after dropping duplicate patient encounters', '\n')

    # Remove direct correlation
    df.drop(['encounter_id', 'patient_nbr'], axis=1, inplace=True)
    print('removed features to prevent perfect label correlation:',
          [i for i in original_columns if i not in df.columns], '\n')

    # Drop medication features as its too sparse (keeps insulin)
    df.drop(df.columns[list(range(22, 39)) + list(range(40, 45))], axis=1, inplace=True)

    # Relabel missing to NaN
    df.replace('?', np.NaN, inplace=True)
    df.replace('Unknown/Invalid', np.NaN, inplace=True)
    df['admission_type_id'].replace([5, 6, 8], np.NaN, inplace=True)
    df['discharge_disposition_id'].replace([18, 25, 26], np.NaN, inplace=True)
    df['admission_source_id'].replace([9, 15, 17, 20, 21], np.NaN, inplace=True)

    # Calculates number of missing cells in each column
    print('percent of entries missing in columns:')
    for i in df.columns:
        if df[i].isna().sum() != 0:
            print(i, round(df[i].isna().sum() * 100 / len(df), 2))
    print('')

    original_columns = df.columns

    # Drop missing columns with 25% or more missing data
    df.dropna(axis=1, thresh=df.shape[0] * missing_percent_threshold, inplace=True)
    print('removed features for having too much missing data:', [i for i in original_columns if i not in df.columns], '\n')

    # Drop missing rows
    df.dropna(axis=0, inplace=True)
    print('dataframe has shape', df.shape, 'after cleaning dataframe', '\n')

    return df


def bin_features(df):
    """Bins features with many values using the dicts from 'dict_mappings.json'"""

    # Dictionary created from the python file 'Create Mapping Dicts'
    with open('dict_mappings.json') as f:
        dict_list = json.load(f)

        # Manual binning from https://www.hindawi.com/journals/bmri/2014/781670/tab2/
        for i in ['diag_1', 'diag_2', 'diag_3']:
            diag_dict = {k: i + '.' + v for k, v in dict_list[0].items()}
            df[i].replace(diag_dict, inplace=True)

        # Manual binning from 'IDs_mapping.csv' file
        admission_type_dict = {int(k): v for k, v in dict_list[1].items()}
        df['admission_type_id'].replace(admission_type_dict, inplace=True)

        # Manual binning from https://www.hindawi.com/journals/bmri/2014/781670/tab3/
        admission_source_dict = {int(k): v for k, v in dict_list[2].items()}
        df['admission_source_id'].replace(admission_source_dict, inplace=True)

        # Manual binning from https://www.hindawi.com/journals/bmri/2014/781670/tab3/
        discharge_dict = {int(k): v for k, v in dict_list[3].items()}
        df['discharge_disposition_id'].replace(discharge_dict, inplace=True)

    return df


def encode_dataset(df):
    """Encode dataset to be used by ML algorithm"""

    # Manually encode some features to keep cross-feature consistency of labeling
    df.replace({'Down': 1, 'Steady': 2, 'Up': 3}, inplace=True)
    df.replace({'No': 0, 'Yes': 1, 'Ch': 1}, inplace=True)
    df.replace({'None': 0, 'Norm': 1, '>7': 2, '>8': 3, '>200': 2, '>300': 3}, inplace=True)
    df.replace({'[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
                '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9})

    # Initialise encoded dataframe
    encoded_df = pd.DataFrame()
    for i in df.columns:

        # If feature is object, create dummy variables and add to new dataframe
        if df[i].dtype == 'object':
            dummies = pd.get_dummies(df[i])
            encoded_df = pd.concat([encoded_df, dummies], axis=1)

        # If feature is already numeric, add to new dataframe
        else:
            encoded_df[i] = df[i]

    return encoded_df


def normalise(df):
    """Normalises df to between 0 and 1"""

    # Retain original columns
    cols = df.columns

    # Normalise dataframe to between 0 and 1
    scaler = MinMaxScaler()
    normalised_df = pd.DataFrame(scaler.fit_transform(df))

    # Add original columns to normalised dataframe
    normalised_df.columns = cols

    return normalised_df


def demographic_analysis(df, protected_groups):
    """Analysis demographic features race, gender and age"""

    # Extract categorical features from dataframe
    categorical_features = [i for i in df.columns if df[i].dtype == 'object']

    # Print the value counts of each protected group
    print(df[protected_groups].value_counts(), '\n')
    # Print the mean and variance of each numeric feature for each protected group
    print(df.groupby(protected_groups).agg(["mean", "var"]), '\n')
    # Print the most common three values for each categorical feature for each protected group
    print(df.groupby(protected_groups)[categorical_features].agg(lambda x: [i for i in x.value_counts().index[:3]]),
          '\n')


def split_dataframe(df, test_size):
    """Extracts label and splits into testing and training"""

    # Extract independent features and label
    X = df.drop('readmitted', axis=1)
    y = df['readmitted']

    # Split dataframe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def balanced_split_dataframe(df, test_size, protected_groups):
    """Splits dataframe into training and testing while retaining protected groups distribution"""

    # Retrieve the value of the protected group that occurs the least
    min_group_count = min([df[i].sum() for i in protected_groups])

    # Initialise new split training and test dataframes
    new_X_train, new_X_test, new_y_train, new_y_test = pd.DataFrame(), pd.DataFrame(), pd.Series(
        dtype='float64'), pd.Series(dtype='float64')
    for i in protected_groups:
        # Extract current group from dataframe and sample the largest amount available to retain protected group balance
        group = df.groupby(i).get_group(1).sample(n=int(min_group_count), random_state=42)
        X_train, X_test, y_train, y_test = split_dataframe(group, test_size)

        # Add sample to new training and test dataframes
        new_X_train = new_X_train.append(X_train)
        new_X_test = new_X_test.append(X_test)
        new_y_train = new_y_train.append(y_train)
        new_y_test = new_y_test.append(y_test)

    return new_X_train, new_X_test, new_y_train, new_y_test


def balanced_accuracy(y, y_pred):
    """Maximises balanced accuracy"""

    # Calculate true positive rate
    true_pos = np.where((y == 1) & (y_pred == 1), 1, 0)
    num_true_pos = np.sum(true_pos)
    true_pos_rate = num_true_pos / np.sum(y == 1)

    # Calculate true negative rate
    true_neg = np.where((y == 0) & (y_pred == 0), 1, 0)
    true_neg_pos = np.sum(true_neg)
    true_neg_rate = true_neg_pos / np.sum(y == 0)

    return (true_pos_rate + true_neg_rate) / 2


def hyperparameter_tuning(X_train, y_train):
    """Performs hyperparameter tuning"""

    # Initialises model and hyperparameters to check
    model = LogisticRegression()
    grid = {
        'max_iter': [1000],
        'random_state': [42],
        'C': [i for i in range(10, 1000, 10)],
        'class_weight': [{0: 1, 1: w / 10} for w in range(10, 200, 10)],
    }

    # Grid search these hyperpameters on our custom scoring metric
    score = make_scorer(balanced_accuracy)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, scoring=score, cv=3)
    tuned_model = grid_search.fit(X_train, y_train)

    # Print and return the best hyperparameters for balanced accuracy
    print("Best: %f using %s" % (tuned_model.best_score_, tuned_model.best_params_))

    return tuned_model.best_params_


def performance_stats(model, X_test, y_test, title):
    """Calculates basic performance stats"""

    y_pred = model.predict(X_test)
    print('performance stats of ' + title + ':')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred), '\n')


def ROC_plot(model, X_test, y_test, title):
    """Plots ROC curve of given model"""

    # Initialise plot
    fig, ax = plt.subplots()
    ax.set_title(title + ' Receiver Operating Characteristic')
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Get ROC and AUC
    logreg_probs = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, logreg_probs)
    fpr, tpr, thresholds = roc_curve(y_test, logreg_probs)

    # Plot ROC and AUC
    ax.plot(fpr, tpr, label='Area Under Curve (AUC = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], 'r--')
    ax.legend(handlelength=0, loc='lower right')

    # Format and save plot
    plt.tight_layout()
    plt.savefig(title + ' ROC Curve.png')


def importance_plot(model, X_test, title):
    """Plots feature importance of given model"""

    # Initialise plot
    fig, ax = plt.subplots()
    ax.set_title(title + ' Feature Importance')
    ax.set_xlabel('Importance')

    # Get and sort the model's feature coefficients
    coef = list(model.coef_[0])
    cols = list(X_test.columns)
    features_df = pd.DataFrame()
    features_df['Features'] = cols
    features_df['importance'] = coef
    features_df.sort_values(by=['importance'], ascending=True, inplace=True)

    # Keep only 10 most and least important features
    features_df = pd.concat([features_df.head(10), features_df.tail(10)])

    # Plot feature importance with negative and positive importance as red and blue respectively
    features_df['positive'] = features_df['importance'] > 0
    features_df.set_index('Features', inplace=True)
    features_df.importance.plot(kind='barh', color=features_df.positive.map({True: 'blue', False: 'red'}), ax=ax)
    ax.set_ylabel('')

    # Format and save plot
    plt.tight_layout()
    plt.savefig(title + ' Feature Importance.png')


def fairness_stats(logreg, X_test, y_test, protected_groups):
    """Calculate logreg predictions on different race groups"""

    # Initialises fairness dataframe
    fair_df = pd.DataFrame(columns=['Protected Group', 'statistical parity difference', 'equal opportunity difference',
                                    'average odds difference'])

    # Appends predicted and actual labels to test dataframe
    test_df = X_test.copy()
    test_df['readmitted'] = y_test

    def get_metrics(group):
        """Calculates various metrics"""

        # Extract X and y for given group
        group_i_df = test_df.groupby(i).get_group(group)
        y_test_i = group_i_df['readmitted']
        X_test_i = group_i_df.drop('readmitted', axis=1)

        # Extract confusion matrix values and calculate metrics
        TN, FP, FN, TP = confusion_matrix(y_test_i, logreg.predict(X_test_i)).ravel()
        PR = (TP + FP) / (TP + FP + FN + TN)
        TPR = (TP) / (TP + FN)
        FPR = (FP) / (FP + TN)

        return PR, TPR, FPR

    for i in protected_groups:
        # Retrieve metrics for privileged and unprivileged groups
        PR1, TPR1, FPR1 = get_metrics(1)  # A = 1
        PR0, TPR0, FPR0 = get_metrics(0)  # A = 0

        # Calculated fairness metrics for protected group
        spd = PR1 - PR0
        eod = TPR1 - TPR0
        aod = (FPR0 - FPR1 + TPR0 - TPR1) / 2

        # Add fairness metrics to fairness dataframe
        fair_df.loc[len(fair_df)] = [i, abs(spd), abs(eod), abs(aod)]

    # Reformat dataframe for plotting
    tidy_df = fair_df.melt(id_vars='Protected Group')

    return tidy_df


def plot_fairness(tidy_df, title):
    """Plots the fairness for each protected feature"""

    # Initialise plot
    fig, ax = plt.subplots()
    ax.set_title(title + ' Fairness Metrics', fontsize=14)
    ax.set(ylim=(0, 0.5))

    # Plot fairness metrics
    sns.barplot(x='Protected Group', y='value', hue='variable', data=tidy_df, palette='Blues', ax=ax)
    ax.legend().set_title('Fairness Metric')
    ax.set_ylabel('')

    # Adds values of columns to the top of relative bars
    for p in ax.patches:
        ax.annotate('{:.2f}'.format(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height() + 0.02),
                    ha='center', va='top', color='black', size=10)

    # Format and save plot
    plt.tight_layout()
    plt.savefig(title + ' Fairness Metrics.png')


def reweighing(X_train, y_train, protected_groups):
    """Produces weights for X_train samples to reduce bias"""

    # Initialise sample weights
    initial_sample_weights = np.ones(X_train.shape[0])
    final_sample_weights = np.empty_like(initial_sample_weights)

    # Calculates the protected groups (as a tuple) and classes
    groups = pd.MultiIndex.from_frame(X_train[protected_groups]).to_flat_index()
    classes = np.unique(y_train)

    def N_sum(i):
        """Returns size of matrix with respect to our initial sample weights (which are actually all 1)"""
        return initial_sample_weights[i].sum()

    for i, g in enumerate(np.unique(groups)):
        for j, c in enumerate(classes):

            # Retrieves boolean array of samples that satisfy current group and class
            g_and_c = (groups == g) & (y_train == c)

            if np.any(g_and_c):
                # Calculates weight factor of current group and class
                W_gc = N_sum(groups == g) * N_sum(y_train == c) / (initial_sample_weights.sum() * N_sum(g_and_c))

                # Assigns new weight to final weight
                final_sample_weights[g_and_c] = W_gc * initial_sample_weights[g_and_c]

    return final_sample_weights


def majority_minority_performance(logreg, X_test, y_test, majority_minority):
    """Prints performance metrics of the majority and minority protected group"""

    # Appends predicted and actual labels to test dataframe
    test_df = X_test.copy()
    test_df['readmitted'] = y_test

    for i in majority_minority:
        # Extract X and y for given group
        group_i_df = test_df.groupby(i).get_group(1)
        y_test_i = group_i_df['readmitted']
        X_test_i = group_i_df.drop('readmitted', axis=1)

        performance_stats(logreg, X_test_i, y_test_i, i + ' Group')


def main(retune_hyperparameters):
    df = pd.read_csv('hospital_readmissons.csv')
    pd.set_option('display.expand_frame_repr', False)
    print('dataframe has shape', df.shape, '\n')

    # 3.2 --------------------------------------------------------------------------------------------------------------
    df = clean_dataset(df, 0.75)
    df = bin_features(df)

    protected_groups = list(df['race'].unique())
    demographic_analysis(df, 'race')

    df = encode_dataset(df)
    df = normalise(df)

    # 3.3 --------------------------------------------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = split_dataframe(df, 0.25)
    print("X_train has shape", X_train.shape)
    print("X_test has shape", X_test.shape, '\n')

    if retune_hyperparameters:
        best_params = hyperparameter_tuning(X_train, y_train)
    else:
        best_params = {'C': 80, 'class_weight': {0: 1, 1: 10}, 'max_iter': 1000, 'random_state': 42}

    tuned_logreg = LogisticRegression(**best_params)
    tuned_logreg.fit(X_train, y_train)
    performance_stats(tuned_logreg, X_test, y_test, 'Tuned Logistic Regression')
    ROC_plot(tuned_logreg, X_test, y_test, 'Tuned Logistic Regression')
    importance_plot(tuned_logreg, X_test, 'Tuned Logistic Regression')
    fair_df = fairness_stats(tuned_logreg, X_test, y_test, protected_groups)
    plot_fairness(fair_df, 'Tuned Logistic Regression')

    balanced_X_train, balanced_X_test, balanced_y_train, balanced_y_test = balanced_split_dataframe(df, 0.25, protected_groups)
    print("balanced X_train has shape", X_train.shape)
    print("balanced X_test has shape", X_test.shape, '\n')

    balanced_logreg = LogisticRegression(**best_params)
    balanced_logreg.fit(balanced_X_train, balanced_y_train)
    performance_stats(balanced_logreg, balanced_X_test, balanced_y_test, 'Balanced Logistic Regression')
    ROC_plot(balanced_logreg, balanced_X_test, balanced_y_test, 'Balanced Logistic Regression')
    importance_plot(balanced_logreg, balanced_X_test, 'Balanced Logistic Regression')
    fair_df = fairness_stats(balanced_logreg, balanced_X_test, balanced_y_test, protected_groups)
    plot_fairness(fair_df, 'Balanced Logistic Regression')

    # 3.4 --------------------------------------------------------------------------------------------------------------
    sample_weight = reweighing(X_train, y_train, protected_groups)

    reweighed_logreg = LogisticRegression(**best_params)
    reweighed_logreg.fit(X_train, y_train, sample_weight)
    performance_stats(reweighed_logreg, X_test, y_test, 'Reweighed Logistic Regression')
    ROC_plot(reweighed_logreg, X_test, y_test, 'Reweighed Logistic Regression')
    importance_plot(reweighed_logreg, X_test, 'Reweighed Logistic Regression')
    fair_df = fairness_stats(reweighed_logreg, X_test, y_test, protected_groups)
    plot_fairness(fair_df, 'Reweighed Logistic Regression')

    majority_minority_performance(reweighed_logreg, X_test, y_test, ['Caucasian', 'Asian'])

    plt.show()


main(False)  # I recommend you don't change retune_hyperparameters to True; it takes a very long time

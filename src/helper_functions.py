import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


def get_columns_to_drop ():
    columns = []

    with open('./text/columns_to_drop.txt') as f:
        for column in f.readlines():
            columns.append (column.strip().replace ('\n', ''))

    return columns


def get_categorical_columns ():
    columns = []

    with open('./text/categorical_columns.txt') as f:
        for column in f.readlines():
            columns.append (column.strip().replace ('\n', ''))

    return columns


def get_target_variables ():
    variables = []

    with open('./text/variables.txt') as f:
        for variable in f.readlines():
            variables.append(variable.strip().replace('\n', ''))

    return variables


def drop_columns (data):
    columns = get_columns_to_drop()
    data.drop (columns, axis = 1, inplace = True)

    return data


def drop_other_target_variables (data, target):
    variables_to_drop = list (
        set (get_target_variables())
        - set ([target])
    )

    return data.drop (variables_to_drop, axis = 1)


def one_hot_encode (data):
    categorical_columns = get_categorical_columns()
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    return data


def do_train_test_split (data, target, proportion = 0.3):
    data = drop_other_target_variables(data, target)
    data = one_hot_encode(data)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop (target, axis=1),
        data [target],
        test_size=proportion,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def eval_metrics(actual, pred):
    acc = accuracy_score(actual, pred)
    prec = precision_score(actual, pred)
    rec = recall_score(actual, pred)
    f1  = f1_score(actual, pred)

    return acc, prec, rec, f1


def pickle_model (model, model_name = 'model'):
    pickle.dump (model, open (f'../models/{model_name}.pkl', 'wb'))
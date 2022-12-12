from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


def do_train_test_split (data, target, proportion = 0.3):
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
# pre-processing
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from utils.utils import *

# model training and evaluation
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from utils.match_rules import match_job_type_rules

# serialize models
import pickle


# global variables
RAND_SEED = 42
TRAIN_FIELD = 'job_type_name'
TRAIN_FRACTION = 0.67
JOB_TYPES = None
N_RAND_SEARCH = 20


# prepare data
def get_dataFrame(file_name='data_with_category.csv', data_path='../data/'):
    return pd.read_csv(data_path + file_name)


def get_desc_token_added(df):
    df['desc_tokenized'] = df.apply(lambda row: tokenize_raw_html(
        str(row['description']) + row['title']), axis=1)
    return df.sample(frac=1, random_state=RAND_SEED)


def get_top_frequent_types(df, n=10):
    type_counter = Counter(df['job_type_name'].tolist())
    top_types = {name for name, _ in type_counter.most_common(n)}
    return df[df['job_type_name'].map(lambda x: x in top_types)]


def get_top_frequent_categories(df, n=5):
    category_counter = Counter(df['job_type_category_name'].tolist())
    top_categories = {name for name, _ in category_counter.most_common(n)}
    return df[df['job_type_name'].map(lambda x: x in top_categories)]


def set_job_types(job_types):
    global JOB_TYPES
    JOB_TYPES = job_types


def get_X_y(df, return_vectorizor=False):
    tfIdfVectorizer = TfidfVectorizer(
        analyzer='word',
        sublinear_tf=True,
        stop_words=STOP_WORDS,
        strip_accents='unicode',
        token_pattern=r'\w{2,}',
        ngram_range=(1, 1),
        max_features=1000,
        use_idf=True
    )
    tfIdf = tfIdfVectorizer.fit(list(df['desc_tokenized']))
    X = tfIdf.transform(list(df['desc_tokenized'])).toarray()
    job_types, y = np.unique(df[TRAIN_FIELD], return_inverse=True)
    set_job_types(job_types)
    if return_vectorizor:
        return X, y, tfIdf
    return X, y


def get_train_test(X, y, train_size=TRAIN_FRACTION):
    n_train = int(len(X) * train_size)
    Xtr, Xts = X[:n_train, :], X[n_train:, :]
    ytr, yts = y[:n_train], y[n_train:]
    return Xtr, Xts, ytr, yts


def get_opt_model_by_grid_search(clf, parameters, Xtr, Xts, ytr, yts, **kwagrs):
    models = RandomizedSearchCV(
        clf, parameters, n_iter=N_RAND_SEARCH, random_state=RAND_SEED)
    models.fit(Xtr, ytr)
    model = models.best_estimator_
    yhat = model.predict(Xts)
    print_report(yts, yhat, **kwagrs)
    return model, yhat


def print_report(yts, yhat, target_names=JOB_TYPES, n_digits=3):
    report = classification_report(
        yts, yhat, labels=[i for i in range(len(target_names))], target_names=target_names, digits=n_digits)
    print(report)


def get_metrics_with_rules(df, Xts, yts, yhat):
    c = Counter()
    job_type_2_class_id = {job_type: i for i, job_type in enumerate(JOB_TYPES)}
    n_train = int(len(df) * TRAIN_FRACTION)

    for i, v in enumerate(Xts):
        title = df.iloc[n_train+i]['title']
        matched_type = match_job_type_rules(title.lower())
        if matched_type:
            if yhat[i] != job_type_2_class_id[matched_type]:
                c[matched_type] += 1
                yhat[i] = job_type_2_class_id[matched_type]

    print('prediction metrics after applying manual rules')
    print_report(yts, yhat)

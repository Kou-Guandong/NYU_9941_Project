import re
import html
import unicodedata
import pickle
import numpy as np
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from utils.match_rules import match_job_type_rules, match_job_exp_rules


def get_model(file_name):
    return pickle.load(open('models/' + file_name, 'rb'))


with open('static/stopwords.txt') as f:
    STOP_WORDS = [line.strip() for line in f.readlines()]

job_relevance_model = get_model('job_relevance_model.pkl')
job_category_model = get_model('job_type_category_name_model.pkl')
job_type_model = get_model('job_type_name_model.pkl')
job_exp_model = get_model('job_exp_name_model.pkl')
tfidfVectorizer = get_model('tfIdfVectorizer.pkl')

job_relevances = ['irrelevant', 'relevant']

job_categories = ['Business Operations', 'Data & Analytics',
                  'Finance, Legal & Compliance', 'Product & Design',
                  'Software Engineering']

job_types = ['Back-End Software Engineering', 'Business Development',
             'Business Intelligence & Data Analysis', 'Data Engineering',
             'DevOps & Infrastructure', 'Front-End Software Engineering',
             'Full-Stack Software Engineering', 'Operations & General Business',
             'Product Manager', 'Sales']


job_experiences = ['Entry Level (New Grad)', 'Intern', 'Junior (1-2 years)']

lemmatizer = WordNetLemmatizer()


def tokenize_text(sentence):
    token_words = word_tokenize(sentence)
    return " ".join([lemmatizer.lemmatize(word.lower(), pos='v') for word in token_words])


def get_text_from_html(raw_html):
    soup = BeautifulSoup(html.unescape(raw_html), features="html.parser")
    text = unicodedata.normalize("NFKD", soup.get_text())
    return text.replace('\n', ' ').replace('\r', '')


def tokenize_raw_html(raw_html):
    return tokenize_text(get_text_from_html(raw_html))


def wrap_result(names, probs):
    return sorted(
        filter(
            lambda x: x['probability'] > 0,
            [{'name': names[i], 'probability': float(
                '%.3f' % p)} for i, p in enumerate(probs)]
        ), key=lambda x: -x['probability'])


def get_result_with_manual_rules(X, model, result_types, func_rule, job_title, job_desc=''):
    # match with manual rules with function (func_tule)
    _matched = func_rule(job_title, job_desc)
    if _matched:
        _result = wrap_result([_matched], [1])
    else:
        _probabilities = model.predict_proba(X).flatten()
        _result = wrap_result(result_types, _probabilities)
    return _result


def predict(job_title, job_desc):
    text_input = tokenize_raw_html(job_title + job_desc)
    X = tfidfVectorizer.transform(np.array([text_input])).toarray()

    # job type
    result_type = get_result_with_manual_rules(
        X, job_type_model, job_types, match_job_type_rules, job_title)

    # job experience
    result_exp = get_result_with_manual_rules(
        X, job_exp_model, job_experiences, match_job_exp_rules, job_title)

    # job category
    prob_category = job_category_model.predict_proba(X).flatten()
    result_category = wrap_result(job_categories, prob_category)

    # job relevance
    prob_relevance = job_relevance_model.predict_proba(X).flatten()
    result_relevance = wrap_result(job_relevances, prob_relevance)

    return {
        'job_category': result_category,
        'job_type': result_type,
        'job_experience': result_exp,
        'job_relevance': result_relevance
    }

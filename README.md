# 9941 Project (Guandong Kou)

This project is used for the auto-classification of job categories, job types and experience levels for the raw HTML date collected from job board websites. The Google Colab for training the model [can be found here](https://colab.research.google.com/drive/1xrfO8Y0J3-_0aiN8Oi9Nv3WPlmliJwGc?usp=sharing)

Currently, it only supports 5 most frequent job categories and 10 most frequence job types. (The rest need more data for the models to learn well.)

Main work in this repo:
- tokenization of raw HTML using beautifulsoup, NLTK
- feature engineering with TF-IDF approach
- model training with logistic regression, support vector classifier, and neural network
- model performance evaluations
- model serialization, API and front-end webpage development

## Environment
Python 3.8.5


## Install Packages

```shell
pip install --user -r requirements.txt
```

```shell
python setup.py
```

## Run Server
```shell
python server.py
```
Then, open `localhost:5000` for API test.

## directories
`manual_rules`: the configuration files for manually added matching rules
`models`: serialized trained models for predictions
`static`: static resources
`templates`: front-end templates (used for API test etc.)
`train`: used for training models in Google Colab
`utils`: utility functions used for predictions and model training
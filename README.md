# 9941 Project (Guandong Kou)

This project is used for the auto-classification of job categories, job types and experience levels for the raw HTML date collected from job board websites.

Currently, it only supports 5 most frequent job categories and 10 most frequence job types. (The rest need more data for the models to learn well.)

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
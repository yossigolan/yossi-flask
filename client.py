import requests
import pandas as pd
import numpy as np

# Read test set and predictions files
x_test_file_name = 'X_test.csv'
y_pred_file_name = 'preds.csv'

# Site urls (local and heroku
site_url_local_parameters = "http://127.0.0.1:5000/predict_churn"
site_url_heroku_parameters = "https://yossi-flask.herokuapp.com/predict_churn"
site_url_local_json = "http://127.0.0.1:5000/predict_churn_bulk"
site_url_heroku_json = "https://yossi-flask.herokuapp.com/predict_churn_bulk"

predict_lines_df = [2, 0, 3, 77, 72]

df_user = pd.read_csv(x_test_file_name)
y_pred_user = np.loadtxt(y_pred_file_name)


def get_params(df, index_num):
    """
    Helper function - get dataset parameters
    :param df: test data set
    :param index_num: row number in dataset
    :return: a dictionary with parameters names and values
    """
    params_dict = {}
    for i, item in enumerate(df.columns):
        params_dict[item]= df.iloc[index_num, i]
    return params_dict


def get_prediction_parameter(df, predict_lines, site_url):
    """
    Ask for predictions using api parameters calls

    :param df: test data set
    :param predict_lines: list of rows in dataset to predict from
    :param site_url: site url
    :return: Nothing. prints predictions.
    """
    print('Predict using api parameters:\n')
    for i, item in enumerate(predict_lines):
        print(f"Prediction {i} is: ", end='')
        params = get_params(df, item)
        print(requests.get(site_url, params=params).text, end='')
        print(f" and the true value is: {int(y_pred_user[item])}")


def get_prediction_json(df, predict_lines, site_url):
    """
    (BONUS) Ask for predictions using api with json file

    :param df: test dataset
    :param predict_lines: list of rows in dataset to predict
    :param site_url: site url
    :return: Nothing - prints predictions
    """
    print('\nPredict using json file:\n')
    json_file = df_user.iloc[predict_lines].to_json(orient='records')
    json_return = requests.post(site_url, json=json_file).json()
    for i, item in enumerate(json_return):
        item['True result ' + str(i)] = y_pred_user[predict_lines[i]]
    print(json_return)


get_prediction_parameter(df_user, predict_lines_df, site_url_local_parameters)
get_prediction_json(df_user, predict_lines_df, site_url_local_json)
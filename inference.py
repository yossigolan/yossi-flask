import flask
import pandas as pd
import numpy as np
import pickle
from flask import Flask, json
from flask import request
import os

# declare parameters

# Load model, test set and preditions files
model_file_name = 'churn_model.pkl'
x_test_file_name = 'X_test.csv'
y_pred_file_name = 'preds.csv'
loaded_model = pickle.load(open(model_file_name, 'rb'))
x_test = pd.read_csv(x_test_file_name)
y_pred = np.loadtxt(y_pred_file_name)


def validate_prediction(y_prediction, x_test_df):
    """
    Validate model predictions
    :param y_prediction: current model predictions
    :param x_test_df: x_test to predict on
    :return: Nothing. Print whether predictions are valid
    """
    y_pred_new = loaded_model.predict(x_test_df)
    if y_prediction.any() == y_pred_new.astype('int').any():
        print('Both predictions are equal! Validation OK!')
    else:
        print('Error in prediction! Validation Error!')


app = Flask(__name__)

validate_prediction(y_pred, x_test)


def get_parameters(df):
    """
    Get parameters from user
    :param df: test data set
    :return: list of parameters values from user
    """
    params_list = []
    returned_list = []
    for i, item in enumerate(df.columns):
        params_list.append(request.args.get(item))
    returned_list.append(params_list)
    return returned_list


# Using api parameters
@app.route('/')
def mainsite():
    message = "<html><p>Please go to https://yossi-flask.herokuapp.com/predict_churn</p>" + \
              "<p>and add parameters:</p>" + \
              "<p>is_male</p>" + \
              "<p>num_inters</p>" + \
              "<p>late_on_payment</p>" + \
              "<p>age</p>" + \
              "<p>years_in_contract</p>" + \
              "<p>OR use https://yossi-flask.herokuapp.com/predict_churn_bulk:</p>" + \
              "<p>with json file (same parameters). </p></html>"
    return message


# Using api parameters
@app.route('/predict_churn')
def predict():
    # get parameters
    x_test_user = get_parameters(x_test)
    # run model and return prediction
    y_predict = loaded_model.predict(x_test_user)
    return f"{y_predict}"


# (BONUS) Using json files
@app.route('/predict_churn_bulk', methods=['POST'])
def predict_bulk():
    # Read json file
    x_test_user = json.loads(request.get_json())
    # run model
    y_predict = loaded_model.predict(pd.DataFrame(x_test_user))
    # Create predictions json file and return the json file
    json_return = []
    for i, pred in enumerate(y_predict):
        result_str = "result-" + str(i)
        json_return.append({result_str: float(pred)})
    json_file = flask.jsonify(json_return)
    return json_file


if __name__ == '__main__':
    # Heroku provides environment variable 'PORT' that should be listened on by Flask
    port = os.environ.get('PORT')

    if port:
        # 'PORT' variable exists - running on Heroku, listen on external IP and on given by Heroku port
        app.run(host='0.0.0.0', port=int(port))
    else:
        # 'PORT' variable doesn't exist, running not on Heroku, presumabely running locally, run with default
        #   values for Flask (listening only on localhost on default Flask port)
        app.run()

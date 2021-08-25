import flask
import pandas as pd
import numpy as np
import pickle
from flask import Flask, json
from flask import request
import os

# Load model, test set and preditions files
model_file_name = 'churn_model.pkl'
x_test_file_name = 'X_test.csv'
y_pred_file_name = 'preds.csv'
loaded_model = pickle.load(open(model_file_name, 'rb'))
x_test_new = pd.read_csv(x_test_file_name)
y_pred_new = np.loadtxt(y_pred_file_name)

# Test model predictions
y_pred = loaded_model.predict(x_test_new)
if y_pred.any() == y_pred_new.astype('int').any():
    print('Both predictions are equal!')
else:
    print('Error in prediction!')

app = Flask(__name__)


# Using api parameters
@app.route('/')
def mainsite():
    message = "Please go to https://yossi-flask.herokuapp.com/predict_churn\n" + \
              "and add parameters:\n" + \
              "is_male\n" + \
              "num_inters\n" + \
              "late_on_payment\n" + \
              "age\n" + \
              "years_in_contract\n" + \
              "OR use https://yossi-flask.herokuapp.com/predict_churn_bulk:\n" + \
              "with json file (same parameters). "
    return message

# http://127.0.0.1:5000/predict_churn?is_male=1&num_inters=0&late_on_payment=0&age=41&years_in_contract=3.240370349
# Using api parameters
@app.route('/predict_churn')
def predict():
    # get parameters
    is_male = request.args.get('is_male')
    num_inters = request.args.get('num_inters')
    late_on_payment = request.args.get('late_on_payment')
    age = request.args.get('age')
    years_in_contract = request.args.get('years_in_contract')
    x_test_user = [[is_male, num_inters, late_on_payment, age, years_in_contract]]
    # run model and return prediction
    y_pred = loaded_model.predict(x_test_user)
    return f"{y_pred}"


# (BONUS) Using json files
@app.route('/predict_churn_bulk', methods=['POST'])
def predict_bulk():
    # Read json file
    x_test_user = json.loads(request.get_json())
    # run model
    y_pred = loaded_model.predict(pd.DataFrame(x_test_user))
    # Create predictions json file and return the json file
    json_return = []
    for i, pred in enumerate(y_pred):
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

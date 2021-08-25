import requests
import pandas as pd
import numpy as np

# Read test set and predictions files
x_test_file_name = 'X_test.csv'
y_pred_file_name = 'preds.csv'

df_user = pd.read_csv(x_test_file_name)
y_pred_user = np.loadtxt(y_pred_file_name)

predict_lines = [2, 0, 3, 77, 72]

# Ask for predictions using api parameters calls
print('Predict using api parameters:\n')
for i, item in enumerate(predict_lines):
    print(f"Prediction {i} is: ", end='')
    params = {'is_male': df_user.iloc[item, 0], 'num_inters': df_user.iloc[item, 1], 'late_on_payment': df_user.iloc[item, 2],
              'age': df_user.iloc[item, 3], 'years_in_contract': df_user.iloc[item, 4]}
    print(requests.get("http://127.0.0.1:5000/predict_churn", params=params).text, end='')
    print(f" and the true value is: {int(y_pred_user[item])}")

# (BONUS) Ask for predictions using api with json file
print('\nPredict using json file:\n')
json_file = df_user.iloc[predict_lines].to_json(orient='records')
json_return = requests.post("http://127.0.0.1:5000/predict_churn_bulk", json=json_file).json()
for i, item in enumerate(json_return):
    item['True result ' + str(i)] = y_pred_user[predict_lines[i]]
print(json_return)

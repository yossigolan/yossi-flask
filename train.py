import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('cellular_churn_greece.csv')

# Split the data
input_features = df.loc[:, df.columns != 'churned']
target_variable = df.loc[:, ['churned']]

X = input_features
y = target_variable.values[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
# Train the model
clf = RandomForestClassifier(n_estimators=50)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model file
model_file_name = 'churn_model.pkl'
pickle.dump(clf, open(model_file_name, 'wb'))

# Save test set and prediction
x_test_file_name = 'X_test.csv'
y_pred_file_name = 'preds.csv'
X_test.to_csv(index=False, path_or_buf=open(x_test_file_name, 'wb'))
np.savetxt(X=y_pred, fname=y_pred_file_name )

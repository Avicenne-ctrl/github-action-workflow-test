#first line creates the file in the trainer folder

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from xgboost import XGBRegressor
import numpy
import argparse
import os
import joblib
import sys
sys.path.append("..")
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing


parser = argparse.ArgumentParser()
parser.add_argument('--l_r', dest='l_r', default=0.001, type=float, help='learning rate')
parser.add_argument('--n_estimators', dest='n_estimators', default=100, type=int, help='n_estimators')
parser.add_argument( '--max_depth',dest='max_depth',  default=6, type=int, help='max_depth')
parser.add_argument( '--subsample', dest='subsample', default=0.8, type=float, help='subsample')

args = parser.parse_args()


def load_data():
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(
        path="boston_housing.npz", test_split=0.2, seed=113
    )

    columns = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", 
        "TAX", "PTRATIO", "B", "LSTAT"
    ]

    # Convertir les ensembles de données en DataFrames pandas
    return pd.DataFrame(x_train, columns=columns), pd.Series(y_train), pd.DataFrame(x_test, columns=columns), pd.Series(y_test)


x_train, y_train, x_val, y_val = load_data()

xgb = XGBRegressor(args.n_estimators, args.max_depth, args.l_r, args.subsample)    

xgb.fit(x_train, y_train)
pred = xgb.predict(x_val)
mse = mean_squared_error(y_val, pred)

print(f"[INFO] : Model MSE: {mse}")

# Check if a previous model exists
best_model_path = "model/best_model.pkl"
if os.path.exists(best_model_path):
    best_model = joblib.load(best_model_path)
    y_best_pred = best_model.predict(x_val)
    best_mse = mean_squared_error(y_val, y_best_pred)
    print(f"[INFO] : Best Model MSE: {best_mse}")
    
else:
    best_mse = float("inf")

# Update the model if it performs better
if mse < best_mse:
    print("[INFO] : New model is better. Updating the saved model...")
    joblib.dump(xgb, best_model_path)
else:
    print("[INFO] : Current model did not outperform the best model.")
    

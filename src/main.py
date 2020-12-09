import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os
import time


def main():
    start = time.time()

    X, y = read_in_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(y_test)
    print(y_pred)
    print(model.score(X_test,y_test))
    er1 = mean_squared_error(y_test, y_pred)
    print("MSE for LinearRegression: %.4f" % (er1))

    baseline = DummyRegressor()
    baseline.fit(X_train, y_train)
    ydummy = baseline.predict(X_test)
    print(baseline.score(X_test, y_test))
    er2 = mean_squared_error(y_test, ydummy)
    print("MSE for DummyRegressor: %.4f" % (er2))

    print("Diff: %.4f, Ratio: %.4f" % (er1-er2, er1/er2))

    end = time.time()
    print(f'Time elapsed: {end - start}')

    
def read_in_data():
    # read in input
    X = np.genfromtxt('x.csv', dtype=int, delimiter=',', names=None)
    y = np.genfromtxt('y.csv', dtype=int, delimiter=',', names=None)
    return X, y


if __name__ == "__main__":
    main()

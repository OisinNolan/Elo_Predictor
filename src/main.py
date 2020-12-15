from sklearn.linear_model import LinearRegression, Lasso, Ridge
import numpy as np
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def mae_ratio(mae, baseline_mae):
    return 1 - (mae / baseline_mae)


GAMES_LIMIT = 10000
MOVES_LIMIT = 50
CV = 10

# TODO(JC): implement file read from global path.

def main():
    from sklearn.dummy import DummyRegressor
    from sklearn.model_selection import cross_validate

    X = np.genfromtxt("x_%s_%d.csv" % (GAMES_LIMIT, MOVES_LIMIT),
                      dtype=float, delimiter=',', names=None)
    y = np.genfromtxt("y_%s_%d.csv" % (GAMES_LIMIT, MOVES_LIMIT),
                      dtype=int, delimiter=',', names=None)

    #scaler = preprocessing.MinMaxScaler()
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)
    X = PolynomialFeatures(1).fit_transform(X)

    print(X.shape)
    baseline = DummyRegressor()
    cv_results = cross_validate(
        baseline, X, y, cv=CV, scoring='neg_mean_absolute_error')
    baseline_fit_time = np.mean(cv_results['fit_time'])
    print(f'Fit time: {baseline_fit_time}')
    baseline_mae = -np.mean(cv_results['test_score'])
    print(f'MAE: {baseline_mae}')

    lr_model = LinearRegression()
    cv_results = cross_validate(
        lr_model, X, y, cv=CV, scoring='neg_mean_absolute_error')
    lr_fit_time = np.mean(cv_results['fit_time'])
    print(f'Fit time: {lr_fit_time}')
    lr_mae = -np.mean(cv_results['test_score'])
    print(f'MAE: {lr_mae}')
    print(f'MAE Ratio: {mae_ratio(lr_mae, baseline_mae)}')

    # lasso_model = Lasso(alpha=1/1000, max_iter=1000)
    # cv_results = cross_validate(
    #     lasso_model, X, y, cv=CV, scoring='neg_mean_absolute_error')
    # lr_fit_time = np.mean(cv_results['fit_time'])
    # print(f'Fit time: {lr_fit_time}')
    # lr_mae = -np.mean(cv_results['test_score'])
    # print(f'MAE: {lr_mae}')
    # print(f'MAE Ratio: {mae_ratio(lr_mae, baseline_mae)}')

    plt.figure()
    k = CV
    C_vals = [0.001,0.01,0.1,1,10,100]
    means = []
    stds = []
    for Ci in C_vals:
      model = Ridge(alpha=1/(Ci), max_iter=10000)
      kf = KFold(n_splits=k)
      predictions = []
      
      for train, test in kf.split(X):
        model.fit(X[train], y[train])
        ypred = model.predict(X[test])
        error = -1*mean_absolute_error(y[test], ypred)
        predictions.append(error)
      means.append(np.array(predictions).mean())
      stds.append(np.array(predictions).std())
    print("Part 2")
    print("Means: %s" % (means))
    print("Std errors: %s" % (stds))
    plt.errorbar(np.array(C_vals), np.array(means), np.array(stds), linestyle=None,
             marker='x', capsize=0)
    plt.xlabel("Ci")
    plt.ylabel("MAE")
    plt.xscale("log")
    # alpha = 1/29
    ridge_model = Ridge(alpha=1/(0.01), max_iter=10000)
    cv_results = cross_validate(
        ridge_model, X, y, cv=CV, scoring='neg_mean_absolute_error')
    lr_fit_time = np.mean(cv_results['fit_time'])
    print(f'Fit time: {lr_fit_time}')
    lr_mae = -np.mean(cv_results['test_score'])
    print(f'MAE: {lr_mae}')
    print(f'MAE Ratio: {mae_ratio(lr_mae, baseline_mae)}')

    plt.show()


if __name__ == "__main__":
    main()

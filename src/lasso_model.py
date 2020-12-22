

from sklearn.linear_model import LinearRegression, Lasso, Ridge
#from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, Normalizer
from sklearn import preprocessing
from features import get_short_features
import testbench
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_validate
import time

GAMES_LIMIT = 50000
MOVES_LIMIT = 30

def games_to_short_features(games):
  return get_short_features(games, GAMES_LIMIT, MOVES_LIMIT)

def test():
  #testbench(knn_pipe, 10000, 1000, 'game2vec_knn_report')
  extractor = FunctionTransformer(games_to_short_features)
  lasso_model = Lasso(alpha=1/(1), max_iter=10000)
  #ridge_model = Lasso(alpha=1/(1), max_iter=1000)
  #ridge_model = DummyRegressor(strategy='median')
  #ridge_model = svm.SVR()
  poly_trans = preprocessing.PolynomialFeatures(degree=2, interaction_only=True)
  lasso_pipe = Pipeline([
    #('Games to short features', extractor),
    ('scaler', StandardScaler()),
    #('scaler', Normalizer()),
    ('Poly features', poly_trans),
    ('Lasso', lasso_model)
  ])
  testbench.test_cached_features(lasso_pipe, 50000, 10000, 'lasso_test_report', depth=MOVES_LIMIT)

def baseline():
  model = DummyRegressor(strategy='mean')
  lasso_pipe = Pipeline([
    #('Games to short features', extractor),
    ('scaler', StandardScaler()),
    ('Model', model)
  ])
  testbench.test_cached_features(lasso_pipe, 50000, 10000, 'lasso_test_report', depth=MOVES_LIMIT)

def feature_selection():
  global GAMES_LIMIT, MOVES_LIMIT
  GAMES_LIMIT = 50000
  MOVES_LIMIT = 20
  X = np.genfromtxt("data/x/short_features/std_x_%s_%d.csv" % (GAMES_LIMIT, MOVES_LIMIT),
                        dtype=float, delimiter=',', names=None)

  X_fil = filter_low_var(X)
  print(X.shape)
  print(X)
  print(X_fil.shape)
  print(X_fil)




def main():
  plt.rc('font', size=14)

  start = time.time()
  print(start)

  test()
  baseline()
  end = time.time()
  print(f'Time elapsed: {end - start}')

if __name__ == "__main__":
    main()

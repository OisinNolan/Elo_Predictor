

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
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
MOVES_LIMIT = 50

def games_to_short_features(games):
  return get_short_features(games, GAMES_LIMIT, MOVES_LIMIT)

def test():
  #testbench(knn_pipe, 10000, 1000, 'game2vec_knn_report')
  extractor = FunctionTransformer(games_to_short_features)
  en_model = ElasticNet(alpha=1/(10), max_iter=10000, l1_ratio= 0.5)
  #ridge_model = Lasso(alpha=1/(1), max_iter=1000)
  #ridge_model = DummyRegressor(strategy='median')
  #ridge_model = svm.SVR()
  poly_trans = preprocessing.PolynomialFeatures(degree=2, interaction_only=True)
  en_pipe = Pipeline([
    #('Games to short features', extractor),
    ('scaler', StandardScaler()),
    #('scaler', Normalizer()),
    ('Poly features', poly_trans),
    ('ElasticNet', en_model)
  ])
  testbench.test_cached_features(en_pipe, 50000, 10000, 'elastic_test_report', depth=MOVES_LIMIT)

def baseline():
  model = DummyRegressor(strategy='mean')
  en_pipe = Pipeline([
    #('Games to short features', extractor),
    ('scaler', StandardScaler()),
    ('Model', model)
  ])
  testbench.test_cached_features(en_pipe, 50000, 10000, 'elastic_test_report', depth=MOVES_LIMIT)





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

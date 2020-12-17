from sklearn.linear_model import LinearRegression, Lasso, Ridge
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from features import get_short_features
import testbench

GAMES_LIMIT = 10000
MOVES_LIMIT = 50

def games_to_short_features(games):
  return get_short_features(games, GAMES_LIMIT, MOVES_LIMIT)

def main():
  #testbench(knn_pipe, 10000, 1000, 'game2vec_knn_report')
  ridge_model = Ridge(alpha=1/(29), max_iter=10000)
  ridge_pipe = Pipeline([
    ('Games to short features', FunctionTransformer(games_to_short_features)),
    ('Ridge', ridge_model)
  ])
  testbench.test(ridge_pipe, 10000, 1000, 'ridge_29_report')

if __name__ == "__main__":
    main()

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
  lasso_model = Lasso(alpha=1/(29), max_iter=10000)
  lasso_pipe = Pipeline([
    ('Games to short features', FunctionTransformer(games_to_short_features)),
    ('Lasso', lasso_model)
  ])
  testbench.test(lasso_pipe, 10000, 1000, 'lasso_29_report')

if __name__ == "__main__":
    main()

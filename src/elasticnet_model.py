from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from features import get_short_features
import testbench

GAMES_LIMIT = 10000
MOVES_LIMIT = 50

def games_to_short_features(games):
  return get_short_features(games, GAMES_LIMIT, MOVES_LIMIT)

ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def main():

  for ratio in ratios:

      elasti_model = ElasticNet(alpha=1/(29), l1_ratio =ratio, max_iter=10000)
      elasti_pipe = Pipeline([
        ('Games to short features', FunctionTransformer(games_to_short_features)),
        ('ElasticNet', elasti_model)
      ])
      testbench.test(elasti_pipe, 10000, 1000, 'elasticNet_report')

if __name__ == "__main__":
    main()

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn import preprocessing
from features import get_short_features
import testbench
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_validate

GAMES_LIMIT = 50000
MOVES_LIMIT = 50

def games_to_short_features(games):
  return get_short_features(games, GAMES_LIMIT, MOVES_LIMIT)

def test():
  #testbench(knn_pipe, 10000, 1000, 'game2vec_knn_report')
  ridge_model = Ridge(alpha=1/(29), max_iter=10000)
  ridge_pipe = Pipeline([
    ('Games to short features', FunctionTransformer(games_to_short_features)),
    ('Ridge', ridge_model)
  ])
  testbench.test(ridge_pipe, 10000, 1000, 'ridge_29_report')

def baseline():
  pass

def cross_val():
  #C_range = [0.001,0.01,0.1,1,10,100,1000]
  C_range = [1,10,20,30,40]
  q_range = [1,2]
  moves_range = [10,20,30,50,100]
  CV = 10 # 10-fold cross-validation
  global GAMES_LIMIT
  GAMES_LIMIT = 50000
  global MOVES_LIMIT

  scores = np.zeros((len(moves_range), len(C_range)))
  errors = np.zeros((len(moves_range), len(C_range)))
  i = 0; j = 0;
  for moves_limit in moves_range:
    MOVES_LIMIT = moves_limit
    X = np.genfromtxt("data/x/short_features/std_x_%s_%d.csv" % (GAMES_LIMIT, MOVES_LIMIT),
                        dtype=float, delimiter=',', names=None)
    y = np.genfromtxt("data/y/short_features/std_y_%s_%d.csv" % (GAMES_LIMIT, MOVES_LIMIT),
                        dtype=int, delimiter=',', names=None)
    
    scaler = preprocessing.MinMaxScaler()
    std_scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)
    X = preprocessing.PolynomialFeatures(1).fit_transform(X)
    j = 0
    for Ci in C_range:
      model = Ridge(alpha=1/(Ci), max_iter=10000)
      cv_results = cross_validate(model, X, y, cv=CV, scoring='r2')
      scores[i,j] = np.mean(cv_results['test_score'])
      errors[i,j] = np.std(cv_results['test_score'])
      j+=1
    i+=1

  # plot the results
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.set_xlabel('moves_limit', fontsize='x-large', c='black', fontweight = 'demi')
  ax.set_ylabel('C', fontsize='x-large', c='black', fontweight = 'demi')
  ax.set_zlabel('r2', fontsize='x-large', c='black', fontweight = 'demi')
  # Make data.
  xx, yy = np.meshgrid(moves_range, C_range)

  print(xx.shape)
  print(yy.shape)
  print(scores.shape)

  print(scores)
  print(errors)

  # Plot the surface.
  surf = ax.plot_surface(xx, yy, np.transpose(scores), cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

  # plot the error bars.
  for i in np.arange(0, len(moves_range)):
    for j in np.arange(0, len(C_range)):
      ax.plot([moves_range[i], moves_range[i]],
       [C_range[j], C_range[j]], [scores[i,j]+errors[i,j], scores[i,j]-errors[i,j]],
        marker="_", c='black',zorder=3)

  # Customize the z axis.
  ax.set_zlim(0.13, 0.15)
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

  # Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=5)

  plt.show()


def main():
  cross_val()

if __name__ == "__main__":
    main()

from sklearn.linear_model import LinearRegression, Lasso, Ridge
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
  global MOVES_LIMIT
  MOVES_LIMIT = 30
  #testbench(knn_pipe, 10000, 1000, 'game2vec_knn_report')
  linear_model = LinearRegression(n_jobs=2)
  scaler = StandardScaler()
  linear_pipe = Pipeline([
    #('Games to short features', FunctionTransformer(games_to_short_features)),
    ('Scaler', scaler),
    ('LinearREgression', linear_model)
  ])
  testbench.test_cached_features(linear_pipe, 40000, 10000, 'lin_reg_test_report', depth=MOVES_LIMIT)

def cross_val():
  q_range = [1,2]
  moves_range = [10,20,30,50,100]
  CV = 10 # 10-fold cross-validation
  global GAMES_LIMIT
  GAMES_LIMIT = 50000
  global MOVES_LIMIT

  scores = np.zeros((len(moves_range), len(q_range)))
  errors = np.zeros((len(moves_range), len(q_range)))
  i = 0; j = 0;
  for moves_limit in moves_range:
    print("i=%d" % (i))
    MOVES_LIMIT = moves_limit
    X = np.genfromtxt("data/x/short_features/std_x_%s_%d.csv" % (GAMES_LIMIT, MOVES_LIMIT),
                        dtype=float, delimiter=',', names=None)
    y = np.genfromtxt("data/y/short_features/std_y_%s_%d.csv" % (GAMES_LIMIT, MOVES_LIMIT),
                        dtype=int, delimiter=',', names=None)
    
    X = preprocessing.PolynomialFeatures(1).fit_transform(X)
    j = 0
    for q in q_range:
      print("j=%d" % (j))
      X = preprocessing.PolynomialFeatures(q, interaction_only=True).fit_transform(X)
      model = LinearRegression()
      pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('Ridge', model)
      ])
      cv_results = cross_validate(pipe, X, y, cv=CV, scoring='r2')
      scores[i,j] = np.mean(cv_results['test_score'])
      errors[i,j] = np.std(cv_results['test_score'])
      j+=1
    i+=1

  print(scores)
  print(errors)

  # plot 2d plots
  # For Q=1, the best is moves=30
  for i in range(len(q_range)):
    plt.figure()
    plt.errorbar(np.array(moves_range), np.array(scores[:,i]), np.array(errors[:,i]),
    linestyle=None, marker='x', capsize=0)
    plt.xlabel("Moves limit")
    plt.ylabel("R2")
    plt.title("q=%d" % (q_range[i]))
    #plt.xscale("log")

  

  # plot the results
  # fig = plt.figure()
  # ax = fig.gca(projection='3d')
  # ax.set_xlabel('moves_limit', fontsize='x-large', c='black', fontweight = 'demi')
  # ax.set_ylabel('q', fontsize='x-large', c='black', fontweight = 'demi')
  # ax.set_zlabel('r2', fontsize='x-large', c='black', fontweight = 'demi')
  # # Make data.
  # xx, yy = np.meshgrid(moves_range, q_range)

  # print(xx.shape)
  # print(yy.shape)
  # print(scores.shape)

  # # Plot the surface.
  # surf = ax.plot_surface(xx, yy, np.transpose(scores), cmap=cm.coolwarm,
  #                       linewidth=0, antialiased=False)

  # # plot the error bars.
  # # for i in np.arange(0, len(moves_range)):
  # #   for j in np.arange(0, len(C_range)):
  # #     ax.plot([moves_range[i], moves_range[i]],
  # #      [C_range[j], C_range[j]], [scores[i,j]+errors[i,j], scores[i,j]-errors[i,j]],
  # #       marker="_", c='black',zorder=3)

  # # Customize the z axis.
  # ax.set_zlim(0.0, 0.30)
  # ax.zaxis.set_major_locator(LinearLocator(10))
  # ax.zaxis.set_major_formatter(FormatStrFormatter('%.03f'))

  # # Add a color bar which maps values to colors.
  # fig.colorbar(surf, shrink=0.5, aspect=5)

  plt.show()

def main():
  plt.rc('font', size=14)
  #cross_val()
  #feature_selection()

  start = time.time()
  test()
  end = time.time()
  print(f'Time elapsed: {end - start}')

if __name__ == "__main__":
    main()

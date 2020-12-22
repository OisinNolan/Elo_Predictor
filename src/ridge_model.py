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
  ridge_model = Ridge(alpha=1/(0.1), max_iter=10000, solver='auto')
  #ridge_model = Lasso(alpha=1/(1), max_iter=1000)
  #ridge_model = DummyRegressor(strategy='median')
  #ridge_model = svm.SVR()
  poly_trans = preprocessing.PolynomialFeatures(degree=2, interaction_only=True)
  ridge_pipe = Pipeline([
    #('Games to short features', extractor),
    ('scaler', StandardScaler()),
    #('scaler', Normalizer()),
    ('Poly features', poly_trans),
    ('Ridge', ridge_model)
  ])
  testbench.test_cached_features(ridge_pipe, 50000, 10000, 'ridge_test_report', depth=MOVES_LIMIT)

def baseline():
  model = DummyRegressor(strategy='mean')
  ridge_pipe = Pipeline([
    #('Games to short features', extractor),
    ('scaler', StandardScaler()),
    ('Model', model)
  ])
  testbench.test_cached_features(ridge_pipe, 50000, 10000, 'ridge_test_report', depth=MOVES_LIMIT)

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


def filter_low_var(X):
  from sklearn.feature_selection import VarianceThreshold
  sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
  sel.fit_transform(X)
  return X

def cross_val_pol(q):
  C_range = [10,20]
  moves_range = [10,20,30,50,100]
  CV = 10 # 10-fold cross-validation

  scores = np.zeros((len(moves_range), len(C_range)))
  errors = np.zeros((len(moves_range), len(C_range)))
  i = 0; j = 0;
  for moves_limit in moves_range:
    print("i=%d" % (i))
    MOVES_LIMIT = moves_limit
    X = np.genfromtxt("data/x/short_features/std_x_%s_%d.csv" % (50000, MOVES_LIMIT),
                        dtype=float, delimiter=',', names=None)
    y = np.genfromtxt("data/y/short_features/std_y_%s_%d.csv" % (50000, MOVES_LIMIT),
                        dtype=int, delimiter=',', names=None)

    X = X[:GAMES_LIMIT, :]
    y = y[:GAMES_LIMIT, :]
    
    X = preprocessing.PolynomialFeatures(q).fit_transform(X)
    j = 0
    for Ci in C_range:
      print("j=%d" % (j))
      model = Ridge(alpha=1/(Ci), max_iter=10000, normalize=False)
      pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('Ridge', model)
      ])
      cv_results = cross_validate(pipe, X, y, cv=CV, scoring='neg_mean_squared_error')
      scores[i,j] = np.mean(cv_results['test_score'])
      errors[i,j] = np.std(cv_results['test_score'])
      j+=1
    i+=1

  # plot the results
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.set_xlabel('moves_limit', fontsize='x-large', c='black', fontweight = 'demi')
  ax.set_ylabel('C', fontsize='x-large', c='black', fontweight = 'demi')
  ax.set_zlabel('-mse', fontsize='x-large', c='black', fontweight = 'demi')
  xx, yy = np.meshgrid(moves_range, C_range)

  print(xx.shape)
  print(yy.shape)
  print(scores.shape)

  print(scores)
  print(errors)

  # Plot the surface.
  surf = ax.plot_surface(xx, yy, np.transpose(scores), cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.show()

def final_cross_val():
  C_range = [30,31,32]
  CV = 10 # 10-fold cross-validation
  global GAMES_LIMIT
  GAMES_LIMIT = 50000
  global MOVES_LIMIT

  scores = np.zeros((len(C_range), 1))
  errors = np.zeros((len(C_range), 1))
  MOVES_LIMIT = 50
  X = np.genfromtxt("data/x/short_features/std_x_%s_%d.csv" % (GAMES_LIMIT, MOVES_LIMIT),
                      dtype=float, delimiter=',', names=None)
  y = np.genfromtxt("data/y/short_features/std_y_%s_%d.csv" % (GAMES_LIMIT, MOVES_LIMIT),
                      dtype=int, delimiter=',', names=None)
  #X = preprocessing.PolynomialFeatures(2).fit_transform(X)
  j = 0
  for Ci in C_range:
    print("j=%d" % (j))
    model = Ridge(alpha=1/(Ci), max_iter=100, normalize=False, solver='saga')
    pipe = Pipeline([
      ('scaler', StandardScaler()),
      ('Poly features', preprocessing.PolynomialFeatures(1)),
      ('Ridge', model)
    ])
    cv_results = cross_validate(pipe, X, y, cv=CV, scoring='neg_mean_squared_error')
    scores[j] = np.mean(cv_results['test_score'])
    errors[j] = np.std(cv_results['test_score'])
    j+=1

  print(scores)
  print(errors)

  # plot 2d plots
  # For Q=1, the best is moves=30
  plt.figure()
  plt.errorbar(np.array(C_range), np.array(scores[:,0]), np.array(errors[:,0]),
  linestyle=None, marker='x', capsize=0)
  plt.xlabel("Ci")
  plt.ylabel("-mse")

  plt.show()


def cross_val():
  #C_range = [0.001,0.01,0.1,1,10,100,1000]
  C_range = [1,10,20,30,40]
  q_range = [1]
  moves_range = [50,100] #[10,20,30,50,100]
  CV = 10 # 10-fold cross-validation
  global GAMES_LIMIT
  GAMES_LIMIT = 50000
  global MOVES_LIMIT

  scores = np.zeros((len(moves_range), len(C_range)))
  errors = np.zeros((len(moves_range), len(C_range)))
  i = 0; j = 0;
  for moves_limit in moves_range:
    print("i=%d" % (i))
    MOVES_LIMIT = moves_limit
    X = np.genfromtxt("data/x/short_features/std_x_%s_%d.csv" % (GAMES_LIMIT, MOVES_LIMIT),
                        dtype=float, delimiter=',', names=None)
    y = np.genfromtxt("data/y/short_features/std_y_%s_%d.csv" % (GAMES_LIMIT, MOVES_LIMIT),
                        dtype=int, delimiter=',', names=None)
    #X = preprocessing.PolynomialFeatures(1).fit_transform(X)
    j = 0
    for Ci in C_range:
      print("j=%d" % (j))
      model = Ridge(alpha=1/(Ci), max_iter=10000, normalize=False)
      pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('Poly features', preprocessing.PolynomialFeatures(1)),
        ('Ridge', model)
      ])
      cv_results = cross_validate(pipe, X, y, cv=CV, scoring='neg_mean_squared_error')
      scores[i,j] = np.mean(cv_results['test_score'])
      errors[i,j] = np.std(cv_results['test_score'])
      j+=1
    i+=1

  # plot the results
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.set_xlabel('moves_limit', fontsize='x-large', c='black', fontweight = 'demi')
  ax.set_ylabel('C', fontsize='x-large', c='black', fontweight = 'demi')
  ax.set_zlabel('-mse', fontsize='x-large', c='black', fontweight = 'demi')
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
  # for i in np.arange(0, len(moves_range)):
  #   for j in np.arange(0, len(C_range)):
  #     ax.plot([moves_range[i], moves_range[i]],
  #      [C_range[j], C_range[j]], [scores[i,j]+errors[i,j], scores[i,j]-errors[i,j]],
  #       marker="_", c='black',zorder=3)

  # Customize the z axis.
  #ax.set_zlim(0.13, 0.15)
  ax.zaxis.set_major_locator(LinearLocator(10))
  #ax.zaxis.set_major_formatter(FormatStrFormatter('%.03f'))

  # Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=5)


  scores = np.zeros((len(moves_range), len(C_range)))
  errors = np.zeros((len(moves_range), len(C_range)))
  i = 0; j = 0;
  for moves_limit in moves_range:
    print("i=%d" % (i))
    MOVES_LIMIT = moves_limit
    X = np.genfromtxt("data/x/short_features/std_x_%s_%d.csv" % (GAMES_LIMIT, MOVES_LIMIT),
                        dtype=float, delimiter=',', names=None)
    y = np.genfromtxt("data/y/short_features/std_y_%s_%d.csv" % (GAMES_LIMIT, MOVES_LIMIT),
                        dtype=int, delimiter=',', names=None)

    #X = preprocessing.PolynomialFeatures(2).fit_transform(X)
    j = 0
    for Ci in C_range:
      print("j=%d" % (j))
      model = Ridge(alpha=1/(Ci), max_iter=10000, normalize=False)
      pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('Poly features', preprocessing.PolynomialFeatures(2)),
        ('Ridge', model)
      ])
      cv_results = cross_validate(pipe, X, y, cv=CV, scoring='neg_mean_squared_error')
      scores[i,j] = np.mean(cv_results['test_score'])
      errors[i,j] = np.std(cv_results['test_score'])
      j+=1
    i+=1

  # plot the results
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.set_xlabel('moves_limit', fontsize='x-large', c='black', fontweight = 'demi')
  ax.set_ylabel('C', fontsize='x-large', c='black', fontweight = 'demi')
  ax.set_zlabel('-mse', fontsize='x-large', c='black', fontweight = 'demi')
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

  ax.zaxis.set_major_locator(LinearLocator(10))



  # scores = np.zeros((len(C_range), len(q_range)))
  # errors = np.zeros((len(C_range), len(q_range)))
  # MOVES_LIMIT = 30 # resuse the knowledge from linear regression
  # X = np.genfromtxt("data/x/short_features/std_x_%s_%d.csv" % (GAMES_LIMIT, MOVES_LIMIT),
  #                       dtype=float, delimiter=',', names=None)
  # y = np.genfromtxt("data/y/short_features/std_y_%s_%d.csv" % (GAMES_LIMIT, MOVES_LIMIT),
  #                       dtype=int, delimiter=',', names=None)                      
  # i = 0; j = 0;
  # for Ci in C_range:
  #   print("i=%d" % (i))
  #   j = 0
  #   for q in q_range:
  #     print("j=%d" % (j))
  #     X = preprocessing.PolynomialFeatures(q).fit_transform(X)
  #     model = Ridge(alpha=1/(Ci), max_iter=10000, normalize=False)
  #     pipe = Pipeline([
  #       ('scaler', StandardScaler()),
  #       ('Ridge', model)
  #     ])
  #     cv_results = cross_validate(pipe, X, y, cv=CV, scoring='neg_mean_squared_error')
  #     scores[i,j] = np.mean(cv_results['test_score'])
  #     errors[i,j] = np.std(cv_results['test_score'])
  #     j+=1
  #   i+=1

  # fig = plt.figure()
  # ax = fig.gca(projection='3d')
  # ax.set_xlabel('C', fontsize='x-large', c='black', fontweight = 'demi')
  # ax.set_ylabel('q', fontsize='x-large', c='black', fontweight = 'demi')
  # ax.set_zlabel('-mse', fontsize='x-large', c='black', fontweight = 'demi')
  # # Make data.
  # xx, yy = np.meshgrid(C_range, q_range)

  # print(xx.shape)
  # print(yy.shape)
  # print(scores.shape)

  # print(scores)
  # print(errors)

  # # Plot the surface.
  # surf = ax.plot_surface(xx, yy, np.transpose(scores), cmap=cm.coolwarm,
  #                       linewidth=0, antialiased=False)
  # # Customize the z axis.
  # #ax.set_zlim(0.13, 0.15)
  # ax.zaxis.set_major_locator(LinearLocator(10))
  # #ax.zaxis.set_major_formatter(FormatStrFormatter('%.03f'))

  # # Add a color bar which maps values to colors.
  # fig.colorbar(surf, shrink=0.5, aspect=5)

  plt.show()


def main():
  plt.rc('font', size=14)
  #cross_val()

  # global GAMES_LIMIT
  #GAMES_LIMIT = 10000
  # cross_val_pol(3)
  
  #feature_selection()

  #final_cross_val()

  start = time.time()
  print(start)
  
  #test()
  baseline()
  end = time.time()
  print(f'Time elapsed: {end - start}')

if __name__ == "__main__":
    main()

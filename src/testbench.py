import chess
import chess.pgn
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')

import json

TRAIN_FILE = 'data/std_train_big.clean.pgn'
TEST_FILE = 'data/std_june.clean.pgn'

def test_cached_features(pipe, train_count, test_count, filename, depth, description=None):
    '''
    Same as test(), but uses pre-processed features stored on disk. This allows for much
    more faster testing.
    '''
    X_train_o = np.genfromtxt("data/x/short_features/std_x_%s_%d.csv" % (50000, depth),
                      dtype=float, delimiter=',', names=None)
    y_train_o = np.genfromtxt("data/y/short_features/std_y_%s_%d.csv" % (50000, depth),
                      dtype=int, delimiter=',', names=None)


    X_train = X_train_o[:train_count,:]
    y_train = y_train_o[:train_count,:]

    fit_start = time.time()
    pipe.fit(X_train, y_train)
    fit_end = time.time()
    fit_time = (fit_end - fit_start)

    y_pred = pipe.predict(X_train)

    train_R2 = r2_score(y_train, y_pred)
    train_MSE = mean_squared_error(y_train, y_pred)

    X_test = np.genfromtxt("data/x/short_features/new_test_std_x_%s_%d.csv" % (12000, depth),
                      dtype=float, delimiter=',', names=None)
    y_test = np.genfromtxt("data/y/short_features/new_test_std_y_%s_%d.csv" % (12000, depth),
                      dtype=int, delimiter=',', names=None)

    X_test = X_test[:test_count,:]
    y_test = y_test[:test_count,:]

    pred_start = time.time()
    y_pred = pipe.predict(X_test)
    pred_end = time.time()
    pred_time = (pred_end - pred_start)

    test_R2 = r2_score(y_test, y_pred)
    test_MSE = mean_squared_error(y_test, y_pred)


    print("ypred: ",y_pred[0], y_pred[1])
    bins = np.linspace(500, 3000, 30)
    plt.hist([y_pred[:,0], y_pred[:,1]], bins, label=['white elo', 'black elo'])
    plt.title("Resulting ELO Predictions")
    plt.xlabel("Predicted ELO rating")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.show()

    results = {
        'Description': description,
        'Pipeline': str(pipe.named_steps),
        '# Games for training': train_count,
        '# Games for testing': test_count,
        'Train Fit time': fit_time,
        'Train R2 score': train_R2,
        'Train MSE': train_MSE,
        'Test Predict time': pred_time,
        'Test R2 score': test_R2,
        'Test MSE': test_MSE
    }
    print(results)
    with open(f'reports/{filename}.json', 'w') as file:
        json.dump(results, file)

def test(pipe, train_count, test_count, filename, description=None):
    '''
    Trains a given pipeline 'pipe' on 'train_count'
    games of chess from TRAIN_FILE and then evaluates it
    on 'test_count' unseen games of chess from TEST_FILE.
    The pipeline is then evaluated and a json report saved in
    reports/<filename>.json
    '''
    train_pgn = open(TRAIN_FILE)
    X_train = []
    y_train = []
    for i in range(train_count):
        game = chess.pgn.read_game(train_pgn)
        if game is not None:
            X_train.append(game)
            y_train.append([int(game.headers['WhiteElo']),int(game.headers['BlackElo'])])

    fit_start = time.time()
    pipe.fit(X_train, y_train)
    fit_end = time.time()
    fit_time = (fit_end - fit_start)

    y_pred = pipe.predict(X_train)

    train_R2 = r2_score(y_train, y_pred)
    train_MSE = mean_squared_error(y_train, y_pred)

    test_pgn = open(TEST_FILE)
    X_test = []
    y_test = []
    for i in range(test_count):
        game = chess.pgn.read_game(train_pgn)
        if game is not None:
            X_test.append(game)
            y_test.append([int(game.headers['WhiteElo']),int(game.headers['BlackElo'])])

    pred_start = time.time()
    y_pred = pipe.predict(X_test)
    pred_end = time.time()
    pred_time = (pred_end - pred_start)

    test_R2 = r2_score(y_test, y_pred)
    test_MSE = mean_squared_error(y_test, y_pred)

    results = {
        'Description': description,
        'Pipeline': str(pipe.named_steps),
        '# Games for training': train_count,
        '# Games for testing': test_count,
        'Train Fit time': fit_time,
        'Train R2 score': train_R2,
        'Train MSE': train_MSE,
        'Test Predict time': pred_time,
        'Test R2 score': test_R2,
        'Test MSE': test_MSE
    }
    print(results)
    with open(f'reports/{filename}.json', 'w') as file:
        json.dump(results, file)

import chess
import chess.pgn
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import json

def test(pipe, train_count, test_count, filename, description=None):
    '''
    Trains a given pipeline 'pipe' on 'train_count'
    games of chess from data/train_50k and then evaluates it
    on 'test_count' unseen games of chess from data/test_10k.

    The pipeline is then evaluated and a json report saved in 
    reports/<filename>.json
    '''
    train_pgn = open('../data/train_50k.pgn')
    X_train = []
    y_train = []
    for i in range(train_count):
        game = chess.pgn.read_game(train_pgn)
        X_train.append(game)
        y_train.append(int(game.headers['WhiteElo']))

    fit_start = time.time()
    pipe.fit(X_train, y_train)
    fit_end = time.time()
    fit_time = (fit_end - fit_start)
    
    test_pgn = open('../data/test_10k.pgn')
    X_test = []
    y_test = []
    for i in range(test_count):
        game = chess.pgn.read_game(train_pgn)
        X_test.append(game)
        y_test.append(int(game.headers['WhiteElo']))
    
    pred_start = time.time()
    y_pred = pipe.predict(X_test)
    pred_end = time.time()
    pred_time = (pred_end - pred_start)

    R2 = r2_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    
    results = {
        'Description': description,
        'Pipeline': str(pipe.named_steps),
        '# Games for training': train_count,
        '# Games for testing': test_count,
        'Fit time': fit_time,
        'Predict time': pred_time,
        'R2 score': R2,
        'MSE': MSE
    }
    
    with open(f'../reports/{filename}.json', 'w') as file:
        json.dump(results, file)
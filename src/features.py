import os
import time
import chess
import chess.pgn
import chess_utils

import numpy as np
from scipy import stats
import pathlib

'''
Reads in GAMES_LIMIT games from the input file and calculates
some very basic features. Outputs the features in x.csv, and
outputs into y.csv.
'''

# Global variables
GAMES_LIMIT = 30000
MOVES_LIMIT = 50
#INPUT_FILE = 'data/fics_202011_notime_50k.pgn'
INPUT_FILE = 'data/fics_202011_notime_50k.pgn'

def store_linear_regression_features():
    start = time.time()
    #pgn = open(f'{pathlib.Path().absolute()}/data/fics_202011_notime_50k.pgn')
    pgn = open(f'../data/fics_202011_notime_50k.pgn')
    games = []
    for i in range(GAMES_LIMIT):
        game = chess.pgn.read_game(pgn)
        games.append(game)

    # features = [white & black piece value, white advantage, advantage variance, num. times advantage changes, result]
    x = np.zeros((GAMES_LIMIT, MOVES_LIMIT*3+3), dtype=int)
    y = np.zeros((GAMES_LIMIT, 1), dtype=int)
    for i in range(GAMES_LIMIT):
        print(i)
        game = games[i]
        if not game:
            break
        x[i, -1] = chess_utils.get_game_result(game, chess.WHITE)
        white_elo = game.headers['WhiteElo']
        y[i, 0] = white_elo
        board = game.board()
        j = 0
        turn = True  # white
        # Get piece value for each move
        for move in game.mainline_moves():
            if(j >= 2*MOVES_LIMIT):
                break
            board.push(move)
            move_val = 0
            if turn:
                move_val = chess_utils.get_piece_value(board, chess.WHITE)
            else:
                move_val = -1*chess_utils.get_piece_value(board, chess.BLACK)
            x[i, j] = move_val
            j += 1
            turn = turn ^ True

        # Get advantage & advantage stats for each move
        board = game.board()
        k = 0
        advantage_diffs = []
        advantage_change_count = 0
        for move in game.mainline_moves():
            if(k >= MOVES_LIMIT):
                break
            board.push(move)
            late_game_limit = 0.66 * int(game.headers['PlyCount'])
            white_advantage = chess_utils.get_board_position_value(board, chess.WHITE, late_game_limit)
            black_advantage = chess_utils.get_board_position_value(board, chess.BLACK, late_game_limit)
            advantage_diff = white_advantage - black_advantage
            x[i, 2*MOVES_LIMIT+k] = advantage_diff
            advantage_diffs.append(advantage_diff)
            if k > 0 and np.sign(advantage_diffs[k]) != np.sign(advantage_diffs[k-1]):
                advantage_change_count += 1
            k += 1
        # Variance & num. times advantage changed
        if len(advantage_diffs):
            x[i, -3] = np.var(advantage_diffs)
        x[i, -2] = advantage_change_count


    np.savetxt("x.csv", x, delimiter=",", fmt='%d')
    np.savetxt("y.csv", y, delimiter=",", fmt='%d')

    end = time.time()
    print(f'Time elapsed: {end - start}')

def clean_data(filename):
    start = time.time()
    #pgn = open(f'{pathlib.Path().absolute()}/data/fics_202011_notime_50k.pgn')
    pgn = open('%s.pgn'%(filename))
    games = []
    i = 0
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        if game.end().ply() < 4:
            continue
        games.append(game)

    f = open("%s.clean.pgn" % (filename), "w")
    for game in games:
        print(game, file=f, end="\n\n")
    f.flush()
    f.close()

    end = time.time()
    print(f'Time elapsed: {end - start}')


def store_short_features():
    # TODO(JC): use absolute paths.

    start = time.time()
    #pgn = open(f'{pathlib.Path().absolute()}/data/fics_202011_notime_50k.pgn')
    pgn = open(INPUT_FILE)
    games = []
    i = 0
    global GAMES_LIMIT
    # GAMES_LIMIT = 10000

    global MOVES_LIMIT
    MOVES_LIMIT = 100

    while i < GAMES_LIMIT:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        if game.end().ply() < 4:
            continue
        games.append(game)
        i+=1

    GAMES_LIMIT = min(GAMES_LIMIT, len(games))
    features_num = 28
    moves_range = [10,20,30,40,50,100]
    x = [np.zeros((GAMES_LIMIT, features_num), dtype=int) for i in moves_range]
    y = np.zeros((GAMES_LIMIT, 2), dtype=int)
    for i in range(len(games)):
        #print(game)
        game = games[i]
        if not game:
            break
        white_elo = game.headers['WhiteElo']
        y[i, 0] = white_elo
        black_elo = game.headers['BlackElo']
        y[i, 1] = black_elo
        board = game.board()
        j = 0
        turn = True  # white
        # Get piece value for each move
        scores = np.zeros(2*MOVES_LIMIT, dtype=float)

        game = games[i]
        for move in game.mainline_moves():
            if(j >= 2*MOVES_LIMIT):
                break
            board.push(move)
            move_val = 0
            if turn:
                move_val = chess_utils.get_board_position_value(board, chess.WHITE, 30)
            else:
                # !!!! NOTE MINUS SIGN HERE
                move_val = -1*chess_utils.get_board_position_value(board, chess.BLACK, 30)
            scores[j] = move_val
            j += 1
            turn = turn ^ True

        k = 0;
        for moves_limit in moves_range: 
            MOVES_LIMIT = moves_limit
            #print(scores)
            (white_feat, black_feat, corr) = scores_to_features(scores[:min(game.end().ply(), 2*moves_limit)])
            x[k][i,0:10] = white_feat
            x[k][i,10:20] = black_feat
            if(np.isnan(corr)):
                # set an arbitrary value
                corr = 3
            x[k][i,20] = corr
            x[k][i,21:28] = game_features(game)
            k += 1
    k = 0;
    for moves_limit in moves_range:
        np.savetxt("data/x/short_features/new_test_std_x_%s_%d.csv" % (GAMES_LIMIT, moves_limit), x[k], delimiter=",", fmt='%.4f')
        np.savetxt("data/y/short_features/new_test_std_y_%s_%d.csv" % (GAMES_LIMIT, moves_limit), y, delimiter=",", fmt='%d')
        k+=1

    end = time.time()
    print(f'Time elapsed: {end - start}')


def get_short_features(games, games_limit, moves_limit):
    global GAMES_LIMIT
    global MOVES_LIMIT
    GAMES_LIMIT = games_limit
    MOVES_LIMIT = moves_limit

    GAMES_LIMIT = min(GAMES_LIMIT, len(games))
    features_num = 28
    x = np.zeros((GAMES_LIMIT, features_num), dtype=int)
    for i in range(len(games)):
        game = games[i]
        board = game.board()
        j = 0
        turn = True  # white
        # Get piece value for each move
        scores = np.zeros(2*MOVES_LIMIT, dtype=float)
        game = games[i]
        for move in game.mainline_moves():
            if(j >= 2*MOVES_LIMIT):
                break
            board.push(move)
            move_val = 0
            if turn:
                move_val = chess_utils.get_board_position_value(board, chess.WHITE, 30)
            else:
                # !!! NOTE MINUS SIGN HERE
                move_val = -1*chess_utils.get_board_position_value(board, chess.BLACK, 30)
            scores[j] = move_val
            j += 1
            turn = turn ^ True
        (white_feat, black_feat, corr) = scores_to_features(scores[:game.end().ply()])
        x[i,0:10] = white_feat
        x[i,10:20] = black_feat
        if(np.isnan(corr)):
            corr = 3
        x[i,20] = corr
        x[i,21:28] = game_features(game)

    return x



def scores_to_features(scores):
    '''
    Input:
        scores - array of scores per each half move.
    Output:
        (white_features, black_features, pearson_corr) :
            a vector of various features for the moves.
            In particular, descriptive statistics.
    '''
    white_moves = scores[::2]
    black_moves  = scores[1::2]
    wsize = white_moves.shape[0]
    bsize = black_moves.shape[0]
    minsize = min(wsize, bsize)
    (corr1_, corr2_) = stats.pearsonr(white_moves[:minsize], black_moves[:minsize])

    return (scores_to_features_1p(white_moves), scores_to_features_1p(black_moves), corr1_)

def scores_to_features_1p(scores):
    '''
        Returns 10 features.
    '''
    # effectively a time series data

    # TODO(JC): the below is probably not very efficient.
    sz = scores.shape[0]
    median_ = np.median(scores)
    mean_ = np.mean(scores)
    half_sz = int(sz/2)
    half1_mean_ = np.mean(scores[:half_sz])
    half2_mean_ = np.mean(scores[half_sz:])
    var_ = np.var(scores)
    min_ = np.amin(scores)
    max_ = np.amax(scores)
    skew_ = stats.skew(scores)
    # aturocorrelation maybe?

    # calculate the largest non-decreasing subarray size.
    inc_streak_ = largest_non_decreasing_len(scores)
    dec_streak_ = largest_non_increasing_len(scores)

    return np.array([median_, mean_, half1_mean_, half2_mean_, var_,
             min_, max_, skew_, inc_streak_, dec_streak_])

def largest_non_decreasing_len(arr):
    res = 0
    l = 0
    n = arr.shape[0]
    for i in range(1, n):
        if(arr[i] >= arr[i-1]):
            l = l + 1
        else:
            res = max(res, l)
            l = 0
    res = max(res, l)
    return res

def largest_non_increasing_len(arr):
    # Can we factor this out?
    res = 0
    l = 0
    n = arr.shape[0]
    for i in range(1, n):
        if(arr[i] <= arr[i-1]):
            l = l + 1
        else:
            res = max(res, l)
            l = 0
    res = max(res, l)
    return res

def encode_result(game):
    '''
    Returns encoding of game result as integer:
        0: white wins (1-0)
        1: black wins (0-1)
        2: draw (1/2-1/2)
    '''
    result = 0
    result_string = game.headers['Result']
    if result_string[0] == '0':
        result = 1
    elif '1/2' in result_string:
        result = 2
    return result

def encode_ending(movetext):
    '''
    Win / Lose:
        0: checkmated
        1: resigned
        2: timeout
    Draw:
        3: repetition
        4: agreement
        5: insufficient material
        6: stalemate
    '''
    ending = 0
    if 'resign' in movetext:
        ending = 1
    elif 'forfeits on time' in movetext:
        ending = 2
    elif 'drawn by repetition' in movetext:
        ending = 3
    elif 'drawn by mutual agreement' in movetext:
        ending = 4
    elif 'Neither player has mating material' in movetext:
        ending = 5
    elif 'drawn by stalemate' in movetext:
        ending = 6
    return ending


def game_features(game):
    '''
    Extracts features from game & game metadata. (7 features)
    '''
    result = encode_result(game)
    num_moves = game.end().ply()
    movetext = str(game.mainline())
    num_checks = movetext.count('+')
    num_kingside_castle = movetext.count('O-O')
    num_queenside_castle = movetext.count('O-O-O')
    num_pawn_promotion = movetext.count('=')
    ending = encode_ending(movetext)
    return [result, num_moves, num_checks, num_kingside_castle, num_queenside_castle, num_pawn_promotion, ending]


def store_game_vec_features():
    '''
    Stores games as a concatenation of board vectors
    '''
    games = np.zeros((GAMES_LIMIT, 64 * MOVES_LIMIT), dtype = int)
    y = np.zeros((GAMES_LIMIT, 1), dtype = int)
    pgn = open(INPUT_FILE)
    for i in range(GAMES_LIMIT):
        game = chess.pgn.read_game(pgn)
        game_as_vec = chess_utils.game_to_vec(game, MOVES_LIMIT)
        games[i] = game_as_vec
        y[i, 0] = white_elo = game.headers['WhiteElo']
    np.save("knn_games", games)
    np.save("knn_y", y)

from sklearn.preprocessing import OneHotEncoder
import pickle

def fit_onehot_encoder():
    '''
    Fits an sklearn OneHotEncoder to moves from GAMES_LIMIT
    number of games. Saves encoder in file 'encoder'
    '''
    all_movetext = []
    pgn = open(INPUT_FILE)
    for i in range(GAMES_LIMIT):
        game = chess.pgn.read_game(pgn)
        movetext = chess_utils.game_to_movetext(game)
        all_movetext = all_movetext + movetext
    all_movetext = np.array(all_movetext).reshape(-1, 1)
    encoder = OneHotEncoder(handle_unknown='ignore').fit(all_movetext)
    with open('encoder', 'wb') as f:
        pickle.dump(encoder, f)

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

def store_text_features():
    '''
    Stores compressed one-hot encodings of PGN movetext
    '''
    encoded_games = []
    elos = []
    pgn = open(INPUT_FILE)
    f = open('encoder', 'rb')
    encoder = pickle.load(f)
    for i in range(GAMES_LIMIT):
        game = chess.pgn.read_game(pgn)
        movetext = chess_utils.game_to_movetext(game, MOVES_LIMIT)
        movetext = np.array(movetext).reshape(-1, 1)
        if not len(movetext):
            continue
        # encode game
        encoding = encoder.transform(movetext).toarray()
        encoding = np.rot90(encoding, axes=(0, 1))
        # compress encoding
        pca = PCA(n_components=MOVES_LIMIT)
        compressed_encoding = pca.fit_transform(encoding)
        compressed_encoding = np.rot90(compressed_encoding, axes=(1, 0))
        encoded_games.append(compressed_encoding.flatten())
        elos.append(game.headers['WhiteElo'])
    np.save('games_text', encoded_games)
    np.save('elos_text', elos)

def main():
    #store_game_vec_features()
    #store_text_features()
    #print(scores_to_features(np.array([1,2,3,4,5,5,6,7,1,2])))
    global MOVES_LIMIT
    global GAMES_LIMIT
    global INPUT_FILE

    clean_data('data/std_june')

    INPUT_FILE = 'data/std_june.clean.pgn'
    GAMES_LIMIT = 12000
    # these are full moves. so *2 for ply moves.
    # Very inefficient, we should really memoise the previous results.
    # for moves_limit in [10,20,30,50,100]: 
    #     MOVES_LIMIT = moves_limit
    store_short_features()

    # clean the data by removing the small games
    # for filename in ['data/std_test_small', 'data/std_train_big']:
    #     clean_data(filename)


if __name__ == '__main__':
    main()

import os
import time
import chess
import chess.pgn
import chess_utils

import numpy as np
import pathlib

'''
Reads in GAMES_LIMIT games from the input file and calculates
some very basic features. Outputs the features in x.csv, and
outputs into y.csv.
'''

# Global variables
GAMES_LIMIT = 10000
MOVES_LIMIT = 10
INPUT_FILE = 'data/fics_202011_notime_50k.pgn'

def store_linear_regression_features():
    start = time.time()
    pgn = open(f'{pathlib.Path().absolute()}/data/fics_202011_notime_50k.pgn')
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

def store_knn_features():
    '''
    Stores game as a concatenation of board vectors
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

def main():
    store_knn_features()

if __name__ == '__main__':
    main()
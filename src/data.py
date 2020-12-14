import os
import time
import chess
import chess.pgn
import chess_utils

import numpy as np

'''
Reads in GAMES_LIMIT games from the input file and calculates
some very basic features. Outputs the features in x.csv, and
outputs into y.csv.
'''

# Global variables
GAMES_LIMIT = 100
MOVES_LIMIT = 15
INPUT_FILE = '../data/fics_202011_notime_50k.pgn'


def store_features():
    start = time.time()
    pgn = open(INPUT_FILE)
    games = []
    for i in range(GAMES_LIMIT):
        game = chess.pgn.read_game(pgn)
        games.append(game)

    x = np.zeros((GAMES_LIMIT, MOVES_LIMIT*2+1), dtype=int)
    y = np.zeros((GAMES_LIMIT, 1), dtype=int)
    for i in range(GAMES_LIMIT):
        game = games[i]
        x[i, -1] = chess_utils.get_game_result(game, chess.WHITE)
        white_elo = game.headers['WhiteElo']
        y[i, 0] = white_elo
        board = game.board()
        j = 0
        turn = True  # white
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

    np.savetxt("x.csv", x, delimiter=",", fmt='%d')
    np.savetxt("y.csv", y, delimiter=",", fmt='%d')

    end = time.time()
    print(f'Time elapsed: {end - start}')

def store_conv():
    start = time.time()
    pgn = open(INPUT_FILE)
    games_limit = 49000
    moves_limit = 50
    games = np.full((games_limit,8,8,moves_limit), 0)
    y = np.zeros((games_limit, 2), dtype=int)
    for i in range(games_limit):
        game = chess.pgn.read_game(pgn)
        if game is None:
            continue; 
        if 'WhiteElo' not in game.headers or 'BlackElo' not in game.headers:
            continue;
        white_elo = game.headers['WhiteElo']
        y[i, 0] = white_elo
        black_elo = game.headers['BlackElo']
        y[i, 1]= black_elo
        board = game.board()
        j = 0
        for move in game.mainline_moves():
            if(j >= moves_limit):
                break
            board.push(move)
            #print(board)
            mat = chess_utils.board_to_mat(board)
            #print(mat)
            games[i,:,:,j] = mat
            j += 1
        
    np.save("mydata2.npy",games)
    np.save("mydata2y.npy",y)
    end = time.time()
    print(f'Time elapsed: {end - start}')

def main():
    #store_features()
    store_conv()



if __name__ == '__main__':
    main()

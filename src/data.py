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
GAMES_LIMIT = 5000
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
    games_limit = 6000
    moves_limit = 70
    games = np.zeros((games_limit,8,8,70))
    y = np.zeros((games_limit, 1), dtype=int)
    for i in range(games_limit):
        game = chess.pgn.read_game(pgn)
        white_elo = game.headers['WhiteElo']
        y[i, 0] = white_elo
        board = game.board()
        j = 0
        for move in game.mainline_moves():
            if(j >= 70):
                break
            board.push(move)
            #print(board)
            mat = chess_utils.board_to_mat(board)
            #print(mat)
            games[i,:,:,j] = mat
            j += 1
        
    np.save("mydata.npy",games)
    np.save("mydatay.npy",y)
    end = time.time()
    print(f'Time elapsed: {end - start}')

def main():
    #store_features()
    store_conv()



if __name__ == '__main__':
    main()

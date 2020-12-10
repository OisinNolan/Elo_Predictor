import chess
import chess.pgn
import numpy as np


def get_game_result(game, color):
    # Given chess.pgn.Game and chess.Color
    # Return -1 if draw, 1 if color won, 0 if color lost.
    if '1/2' in game.headers['Result']:
        return -1
    elif color == chess.WHITE:
        return game.headers['Result'][0]
    else:
        return game.headers['Result'][2]


# [pawn, knight, bishop, rook, queen], see https://en.wikipedia.org/wiki/Chess_piece_relative_value
PIECES_VALUES = [1, 3, 3, 5, 9]


def get_piece_value(board, color):
    # Given chess.Board and chess.Color
    # Returns sum of piece values for that color
    piece_value_sum = 0
    for i in range(0, 5):
        piece_value_sum += PIECES_VALUES[i] * len(board.pieces(i+1, color))
    return piece_value_sum


PIECE_ID = {
    chess.QUEEN: 6,
    chess.ROOK: 5,
    chess.KNIGHT: 4,
    chess.BISHOP: 3,
    chess.PAWN: 2,
    chess.KING: 1
}


def board_to_mat(board):
    '''
      For a given board return a 2d numpy matrix representing this board.
    '''
    mat = np.zeros((8, 8), dtype=int)
    for square in chess.SQUARES:
        i = int(square / 8)
        j = int(square % 8)
        piece = board.piece_at(square)
        if piece is not None:
            id = PIECE_ID[piece.piece_type]
            if piece.color == chess.WHITE:
                mat[i, j] = id
            else:
                mat[i, j] = -1*id
    return mat

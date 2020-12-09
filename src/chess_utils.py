import chess
import chess.pgn

# Given chess.pgn.Game and chess.Color
# Return -1 if draw, 1 if color won, 0 if color lost.
def get_game_result(game, color):
    if '1/2' in game.headers['Result']:
        return -1
    elif color == chess.WHITE:
        return game.headers['Result'][0]
    else:
        return game.headers['Result'][2]


# [pawn, knight, bishop, rook, queen], see https://en.wikipedia.org/wiki/Chess_piece_relative_value
PIECES_VALUES = [1, 3, 3, 5, 9]
# Given chess.Board and chess.Color
# Returns sum of piece values for that color
def get_piece_value(board, color):
    piece_value_sum = 0
    for i in range(0, 5):
        piece_value_sum += PIECES_VALUES[i] * len(board.pieces(i+1, color))
    return piece_value_sum
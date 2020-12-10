import chess
import chess.pgn
import position_value

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

# Functions from notebooks/data_exploration on Anton branch
weights_w = position_value.weights_w
weights_b = position_value.weights_b

# Gets number indicating how strong the positioning of a given piece is
def get_piece_position_value(piece, i, j, moveNum, limit):
    if piece == None:
        return
    weights = []
    if piece.color == chess.WHITE:
        weights = weights_w
    elif piece.color == chess.BLACK:
        weights = weights_b
    else:
        print("Invalid color")
        return 0
    
    pt = piece.piece_type
    if pt == 6:
        if moveNum > limit:
            return weights[pt][i][j]
    return weights[pt-1][i][j]

# Gives number indicating how strong a given player's position is
def get_board_position_value(board, color, limit):
    if color != chess.BLACK and color != chess.WHITE:
        return
    sum_of_weights = 0
    count = 0
    for i in range(7,-1,-1):
        for j in range(7,-1,-1):
            res = 0
            piece = board.piece_at(chess.SQUARES[i * 8 + j])
            moveNum = board.fullmove_number
            if  piece != None and piece.color == color:
                res = get_piece_position_value(piece, i, j, moveNum, limit)
                count += 1
            if res != None:
                sum_of_weights += res
    return sum_of_weights / count
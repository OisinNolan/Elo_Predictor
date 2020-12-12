import chess
import chess.engine
import chess.pgn
import matplotlib.pyplot as plt
import sys
import numpy as np

X = []
y = []

engine = chess.engine.SimpleEngine.popen_uci("stockfish_20090216_x64")

def stockfish_evaluation(board, time_limit = 0.01):
    result = engine.analyse(board, chess.engine.Limit(time=time_limit))
    return result['score']

def blunder(x,y):
    if y > x:
        return False
    else:
        if abs(y - x) > 100:
            return True

def getAUC(advantage):
    return np.trapz(np.array(advantage))


limit = 100
game_count =0
pgn = open('../data/fics_202011_notime_50k.pgn')
game = chess.pgn.read_game(pgn)
while game_count < limit:
    advantage_w = []

    totalMoves = int(game.headers['PlyCount'])
    if totalMoves < 5:
        game = chess.pgn.read_game(pgn)

    white_elo = int(game.headers['WhiteElo'])


    board = game.board()
    for move in game.mainline_moves():
        score_w = stockfish_evaluation(board)
        advantage_w.append(score_w.white().score())
        board.push(move)

    points = []
    for x in advantage_w:
        if x is not None:
            points.append(x)
    '''
    blunderCount = 0
    for i in range(len(points)-1):

        if blunder(points[i], points[i+1]):
            blunderCount += 1

    if totalMoves != 0 and blunderCount > 0:
        y.append(blunderCount/totalMoves)
        X.append(white_elo)

    '''
    metric = getAUC(points)/totalMoves
    y.append(metric)
    X.append(white_elo)

    print("Game parsed Moves : ",totalMoves )
    #print("BlunderCount: ", blunderCount)
    #print("Points: ", points, "\n")
    game = chess.pgn.read_game(pgn)
    game_count += 1


plt.figure(1, figsize=(6, 4))
plt.title("Testing")
plt.scatter(X, y, color = 'blue')
plt.xlabel("ELO")
plt.ylabel("Blunders divided by total moves")
plt.show()

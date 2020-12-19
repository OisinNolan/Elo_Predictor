
import numpy as np
import chess
import chess.pgn
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
import chess_utils

TRAIN_FILE = '../data/std_train_big.clean.pgn'

white_elos = []
black_elos = []
advantage_w = []



GAMES_LIMIT = 10000
MOVES_LIMIT = 50

train_pgn = open(TRAIN_FILE)



for i in range(GAMES_LIMIT):
    game = chess.pgn.read_game(train_pgn)
    white_elos.append(int(game.headers['WhiteElo']))
    black_elos.append(int(game.headers['BlackElo']))

plt.title("ELO Frequency in filtered data")
sns.displot(white_elos, color = 'red')


plt.xlabel("Black ELO")
plt.ylabel("Frequency")
plt.show()

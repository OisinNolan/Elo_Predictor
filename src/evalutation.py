from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
import testbench
import chess_utils
import numpy as np

def games_to_opening_vecs(games):
    return np.array(list(map(lambda game: chess_utils.game_to_vec(game, 10), games)))

knn_model = KNeighborsRegressor(n_neighbors = 12, weights = 'uniform', metric='hamming')
knn_pipe = Pipeline([
    ('Game to vec', FunctionTransformer(games_to_opening_vecs)),
    ('kNN', knn_model)
])

testbench.test(knn_pipe, 10000, 1000, 'game2vec_knn_report')
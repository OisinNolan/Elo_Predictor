from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
import testbench
import chess_utils
import numpy as np

def games_to_opening_vecs(games):
    return np.array(list(map(lambda game: chess_utils.game_to_vec(game, 35), games)))

def gaussian_kernel(distances):
    weights = np.exp(-75 * (distances ** 2))
    return weights / np.sum(weights)

knn_model = KNeighborsRegressor(n_neighbors = 12, weights = gaussian_kernel, metric='hamming')
knn_pipe = Pipeline([
    ('Game to vec', FunctionTransformer(games_to_opening_vecs)),
    ('kNN', knn_model)
])

# 2 hyperparams to tune: # moves considered in game2vec; # neighbours considered

testbench.test(knn_pipe, 100, 10, 'test_knn_report', 'kNN test')
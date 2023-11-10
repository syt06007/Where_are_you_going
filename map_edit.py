import numpy as np

data = np.load('2gong_grid_map.npy')

np.savetxt('map.csv', data, delimiter=',')
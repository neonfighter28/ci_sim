import numpy as np

class Canals:
    def normalize(side):
        return (side.T / np.sqrt(np.sum(side ** 2, axis=1))).T
    right= normalize(
        np.array(
            [[0.32269, -0.03837, -0.94573],
             [0.58930, 0.78839, 0.17655],
             [0.69432, -0.66693, 0.27042]]),)

    left= normalize(
        np.array([[-0.32269, -0.03837, 0.94573],
                [-0.58930, 0.78839, -0.17655],
                [-0.69432, -0.66693, -0.27042]]))

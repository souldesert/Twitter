import numpy as np


def round_(results):
    mean = np.mean(np.asarray(results), axis=0)
    return np.around(mean, decimals=1).tolist()

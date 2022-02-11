import numpy as np


def dice(calculated, reference, k=1):
    intersection = np.sum(calculated[reference == k]) * 2.0
    return intersection / (np.sum(calculated) + np.sum(reference))

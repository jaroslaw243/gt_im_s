import copy
import numpy as np


def dice(calculated, reference, k=1):
    intersection = np.sum(calculated[reference == k]) * 2.0
    return intersection / (np.sum(calculated) + np.sum(reference))


def add_gaussian_noise(image, s_deviation, mean=0):
    if not (s_deviation == 0 and mean == 0):
        gaussian_noise = np.random.normal(mean, s_deviation, size=image.shape)

        img_noise_temp = np.array(image, dtype=np.int32) + gaussian_noise
        img_noise_temp[img_noise_temp > 255] = 255
        img_noise_temp[img_noise_temp < 0] = 0
        img_noise = np.array(img_noise_temp, dtype=np.uint8)
    else:
        img_noise = copy.copy(image)

    return img_noise

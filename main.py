import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('test_circle.png', 0)

gaussian_noise = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
cv2.randn(gaussian_noise, 100, 20)
img_noise = img + gaussian_noise

plt.imshow(img_noise, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()

import cv2
import numpy as np
from matplotlib import pyplot as plt


def region_segmentation_cost(image, segmentation, constant):
    data_fidelity_term = np.sum(np.square(image - segmentation))
    smoothness_term = 0
    segmentation = np.int8(np.pad(segmentation, 1, 'edge'))
    for i in range(1, segmentation.shape[0] - 2):
        for j in range(1, segmentation.shape[1] - 2):
            smoothness_term += (np.square(segmentation[i][j] - segmentation[i - 1][j]) + np.square(segmentation[i][j] - segmentation[i][j + 1]))

    return data_fidelity_term + (constant ** 2) * smoothness_term


img = cv2.imread('test_circle.png', 0)

gaussian_noise = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
cv2.randn(gaussian_noise, 80, 25)
img_noise = img + gaussian_noise

img_gradient = cv2.Laplacian(img_noise, cv2.CV_64F)

ret, best_img_seg = cv2.threshold(img_noise, 100, 1, cv2.THRESH_BINARY)
min_cost = region_segmentation_cost(img_noise, best_img_seg, 2)
worst_img_seg = best_img_seg
max_cost = min_cost
for tr in range(140, 210, 10):
    ret, img_seg = cv2.threshold(img_noise, tr, 1, cv2.THRESH_BINARY)
    cost = region_segmentation_cost(img_noise, img_seg, 2)

    if cost < min_cost:
        min_cost = cost
        best_img_seg = img_seg

    if cost > max_cost:
        max_cost = cost
        worst_img_seg = img_seg

fig, ax = plt.subplots(1, 3)
plt.setp(ax, xticks=[], yticks=[])
ax[0].imshow(img_noise, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(best_img_seg, cmap='gray')
ax[1].set_title(f'Best segmentation (cost {min_cost})')
ax[2].imshow(worst_img_seg, cmap='gray')
ax[2].set_title(f'Worst segmentation (cost {max_cost})')

fig2, ax2 = plt.subplots(1, 2)
plt.setp(ax2, xticks=[], yticks=[])
ax2[0].imshow(img_noise, cmap='gray')
ax2[0].set_title('Original')
ax2[1].imshow(img_gradient, cmap='gray')
ax2[1].set_title('Gradient')

plt.show()

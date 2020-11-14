import cv2
import numpy as np
from matplotlib import pyplot as plt


def region_segmentation_cost(image, segmentation, constant):
    data_fidelity_term = np.sum(np.square(image - segmentation))
    smoothness_term1 = 0
    smoothness_term2 = 0
    segmentation = np.int8(np.pad(segmentation, 1, 'edge'))
    for i in range(1, segmentation.shape[0] - 2):
        for j in range(1, segmentation.shape[1] - 2):
            smoothness_term1 += (segmentation[i][j] - segmentation[i - 1][j]) ** 2
            smoothness_term2 += (segmentation[i][j] - segmentation[i][j + 1]) ** 2

    return data_fidelity_term + (constant ** 2)*(smoothness_term1 + smoothness_term2)


img = cv2.imread('test_circle.png', 0)

gaussian_noise = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
cv2.randn(gaussian_noise, 120, 200)
img_noise = img + gaussian_noise

ret, best_img_seg = cv2.threshold(img_noise, 100, 1, cv2.THRESH_BINARY)
min_cost = region_segmentation_cost(img_noise, best_img_seg, 2)
for tr in range(140, 210, 10):
    ret, img_seg = cv2.threshold(img_noise, tr, 1, cv2.THRESH_BINARY)
    cost = region_segmentation_cost(img_noise, img_seg, 2)

    if cost < min_cost:
        min_cost = cost
        best_img_seg = img_seg

fig, ax = plt.subplots(1, 2)
plt.setp(ax, xticks=[], yticks=[])
ax[0].imshow(img_noise, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(best_img_seg, cmap='gray')
ax[1].set_title(f'Segmentation (cost {min_cost})')
plt.show()

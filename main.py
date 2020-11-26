import cv2
# import numpy as np
import autograd.numpy as np
from autograd import grad
from matplotlib import pyplot as plt
import copy


def region_segmentation_cost(segmentation):
    constant = 2

    img = cv2.imread('test_circle.png', 0)

    gaussian_noise = np.random.normal(0, 25, size=(img.shape[0], img.shape[1]))

    img_noise_temp = np.array(img, dtype=np.int32) + gaussian_noise
    img_noise_temp[img_noise_temp > 255] = 255
    img_noise_temp[img_noise_temp < 0] = 0
    image = np.array(img_noise_temp, dtype=np.uint8)

    data_fidelity_term = np.sum(np.square(image - segmentation))
    smoothness_term = 0
    segmentation = np.pad(segmentation, 1, 'constant')
    for i in range(1, segmentation.shape[0] - 2):
        for j in range(1, segmentation.shape[1] - 2):
            smoothness_term += (np.square(segmentation[i][j] - segmentation[i - 1][j]) + np.square(segmentation[i][j] - segmentation[i][j + 1]))

    return data_fidelity_term + (constant ** 2) * smoothness_term


def boundary_segmentation_cost(image, contour):
    b_cost = 0
    for i in range(len(contour[0])):
        x = contour[0][i][0][0]
        y = contour[0][i][0][1]
        b_cost += image[x, y]
    return b_cost


img = cv2.imread('test_circle.png', 0)

gaussian_noise = np.random.normal(0, 25, size=(img.shape[0], img.shape[1]))

img_noise_temp = np.array(img, dtype=np.int32) + gaussian_noise
img_noise_temp[img_noise_temp > 255] = 255
img_noise_temp[img_noise_temp < 0] = 0
img_noise = np.array(img_noise_temp, dtype=np.uint8)

img_gradient = cv2.Laplacian(img_noise, cv2.CV_64F, ksize=5)

et, img_cn = cv2.threshold(cv2.imread('contour.png', 0), 125, 1, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(img_cn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boundary_cost = boundary_segmentation_cost(img_gradient, contours)

et2, img_cn2 = cv2.threshold(img, 150, 1, cv2.THRESH_BINARY)
contours2, hierarchy2 = cv2.findContours(img_cn2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boundary_cost2 = boundary_segmentation_cost(img_gradient, contours2)

img_contour = copy.copy(img_noise)
cv2.drawContours(img_contour, contours, -1, 255, 3)

img_contour2 = copy.copy(img_noise)
cv2.drawContours(img_contour2, contours2, -1, 255, 3)

ret, best_img_seg = cv2.threshold(img_noise, 100, 1, cv2.THRESH_BINARY)
# min_cost = region_segmentation_cost(img_noise, best_img_seg, 2)
# worst_img_seg = best_img_seg
# max_cost = min_cost
# for tr in range(140, 210, 10):
#     ret, img_seg = cv2.threshold(img_noise, tr, 1, cv2.THRESH_BINARY)
#     cost = region_segmentation_cost(img_noise, img_seg, 2)
#
#     if cost < min_cost:
#         min_cost = cost
#         best_img_seg = img_seg
#
#     if cost > max_cost:
#         max_cost = cost
#         worst_img_seg = img_seg

min_cost = region_segmentation_cost(best_img_seg)
super_jajo = grad(region_segmentation_cost)
worst_img_seg = best_img_seg
max_cost = min_cost
for tr in range(140, 210, 10):
    ret, img_seg = cv2.threshold(img_noise, tr, 1, cv2.THRESH_BINARY)
    best_img_seg -= super_jajo(np.array(img_seg, dtype=np.float))

print(best_img_seg)


fig, ax = plt.subplots(1, 3)
plt.setp(ax, xticks=[], yticks=[])
ax[0].imshow(img_noise, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(best_img_seg, cmap='gray')
ax[1].set_title(f'Best segmentation (cost {min_cost})')
ax[2].imshow(worst_img_seg, cmap='gray')
ax[2].set_title(f'Worst segmentation (cost {max_cost})')

fig2, ax2 = plt.subplots(1, 4)
plt.setp(ax2, xticks=[], yticks=[])
ax2[0].imshow(img_noise, cmap='gray')
ax2[0].set_title('Original')
ax2[1].imshow(img_gradient, cmap='gray')
ax2[1].set_title('Gradient')
ax2[2].imshow(img_contour, cmap='gray')
ax2[2].set_title(f'Contour (cost {boundary_cost})')
ax2[3].imshow(img_contour2, cmap='gray')
ax2[3].set_title(f'Contour (cost {boundary_cost2})')

plt.show()

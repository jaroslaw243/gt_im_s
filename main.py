import cv2
import numpy as np
from autograd import grad
from matplotlib import pyplot as plt
import copy


def region_segmentation_cost(image, segmentation, constant):
    data_fidelity_term = np.sum(np.square(image - segmentation))
    smoothness_term = 0
    segmentation = np.array(np.pad(segmentation, 1, 'constant'), dtype=np.int8)
    for i in range(1, segmentation.shape[0] - 2):
        for j in range(1, segmentation.shape[1] - 2):
            smoothness_term += (np.square(segmentation[i][j] - segmentation[i - 1][j]) + np.square(
                segmentation[i][j] - segmentation[i][j + 1]))

    return data_fidelity_term + (constant ** 2) * smoothness_term


def boundary_segmentation_cost(image, contour):
    b_cost = 0
    for i in range(len(contour[0])):
        x = contour[0][i][0][0]
        y = contour[0][i][0][1]
        b_cost += image[x, y]
    return b_cost


def region_segmentation_cost_clique(image, segmentation, constant, n_size, i, j, change=False):
    segmentation = np.int8(np.pad(segmentation, n_size, 'edge'))
    image = np.int8(np.pad(image, n_size, 'edge'))
    if change:
        if segmentation[i, j] == 1:
            segmentation[i, j] = 0
        else:
            segmentation[i, j] = 1

    data_fidelity_term = np.sum(np.square(image[i-n_size:i+(n_size + 1), j-n_size:j+(n_size + 1)] -
                                          segmentation[i-n_size:i+(n_size + 1), j-n_size:j+(n_size + 1)]))
    smoothness_term = np.sum(np.square(segmentation[i-n_size:i+(n_size + 1),
                                       j-n_size:j+(n_size + 1)] - segmentation[i, j]))
    return data_fidelity_term + (constant ** 2) * smoothness_term


def gradient_descent(image, w1,  neighborhood_size, max_iteration=50):
    w = copy.copy(w1)
    dims = w.shape
    new_w = copy.copy(w)
    for x in range(dims[0]):
        for y in range(dims[1]):
            current_energy = region_segmentation_cost_clique(image, new_w, 20, neighborhood_size, x, y)

            new_energy = region_segmentation_cost_clique(image, new_w, 20, neighborhood_size, x, y, True)

            if new_energy < current_energy:
                if w[x - neighborhood_size, y - neighborhood_size] == 1:
                    w[x - neighborhood_size, y - neighborhood_size] = 0
                else:
                    w[x - neighborhood_size, y - neighborhood_size] = 1

            # if np.abs(new_energy - current_energy) < 1:
            #     return w
    return w


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


ret, best_img_seg = cv2.threshold(img_noise, 180, 1, cv2.THRESH_BINARY)
gd_segmentation = gradient_descent(img_noise, best_img_seg, 1)
min_cost = region_segmentation_cost(img_noise, best_img_seg, 2)


fig, ax = plt.subplots(1, 3)
plt.setp(ax, xticks=[], yticks=[])
ax[0].imshow(img_noise, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(best_img_seg, cmap='gray')
ax[1].set_title('Initial segmentation')
ax[2].imshow(gd_segmentation, cmap='gray')
ax[2].set_title('Gradient descent')

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

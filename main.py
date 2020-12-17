import cv2
# import numpy as np
import autograd.numpy as np
from autograd import grad
from pyefd import elliptic_fourier_descriptors
from matplotlib import pyplot as plt
import copy
import time


def fourier_parametrization_to_indices(coefficients, accuracy):
    k_max = int((coefficients.size - 2) / 4)

    a0_c0 = coefficients[0:2]
    contour = []
    for t in np.arange(0, 2 * np.pi, accuracy):
        second_term = np.zeros((1, 2), dtype=np.float)
        coef_indices = np.array((2, 3, 4, 5), dtype=np.uint16)

        for k in range(1, k_max + 1):
            a_n = coefficients[coef_indices[0]]
            b_n = coefficients[coef_indices[1]]
            c_n = coefficients[coef_indices[2]]
            d_n = coefficients[coef_indices[3]]
            second_term += np.matmul(
                np.array([[a_n, b_n], [c_n, d_n]], dtype=np.float),
                np.array([np.cos(k * t), np.sin(k * t)], dtype=np.float))
            coef_indices += 4
        indices = np.rint(a0_c0 + second_term)

        contour.append(indices)

    return [np.array(contour, dtype=np.int32)]


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
    segmentation_clique = copy.copy(segmentation[i - n_size:i + (n_size + 1), j - n_size:j + (n_size + 1)])
    image_clique = image[i - n_size:i + (n_size + 1), j - n_size:j + (n_size + 1)]
    if change:
        if segmentation_clique[n_size, n_size] == 1:
            segmentation_clique[n_size, n_size] = 0
        else:
            segmentation_clique[n_size, n_size] = 1

    data_fidelity_term = np.sum(np.square(image_clique - segmentation_clique))
    smoothness_term = np.sum(np.square(segmentation_clique - segmentation_clique[n_size, n_size]))
    return data_fidelity_term + (constant ** 2) * smoothness_term


def region_segmentation_cost_clique_interlaced(image, segmentation, constant, contour, alpha, n_size, u, v,
                                               i, j, change=False):
    segmentation_clique = copy.copy(segmentation[i - n_size:i + (n_size + 1), j - n_size:j + (n_size + 1)])
    image_clique = image[i - n_size:i + (n_size + 1), j - n_size:j + (n_size + 1)]
    if change:
        if segmentation_clique[n_size, n_size] == 1:
            segmentation_clique[n_size, n_size] = 0
        else:
            segmentation_clique[n_size, n_size] = 1

    data_fidelity_term = np.sum(np.square(image_clique - segmentation_clique))
    smoothness_term = np.sum(np.square(segmentation_clique - segmentation_clique[n_size, n_size]))

    boundary_seg = contour[i - n_size:i + (n_size + 1), j - n_size:j + (n_size + 1)]

    sum_in = np.sum(np.square(segmentation_clique[boundary_seg == 1] - u))
    sum_out = np.sum(np.square(segmentation_clique[boundary_seg == 0] - v))

    return (data_fidelity_term + (constant ** 2) * smoothness_term) + (alpha * (sum_in + sum_out))


def iterated_conditional_modes(image, w1, neighborhood_size, smoothness_const):
    w = copy.copy(w1)
    dims = w.shape
    new_w = np.array(np.pad(w, neighborhood_size, 'edge'), dtype=np.int32)
    image = np.array(np.pad(image, neighborhood_size, 'edge'), dtype=np.int32)
    for x in range(neighborhood_size, dims[0] + neighborhood_size):
        for y in range(neighborhood_size, dims[1] + neighborhood_size):
            current_energy = region_segmentation_cost_clique(image, new_w, smoothness_const, neighborhood_size, x, y)

            new_energy = region_segmentation_cost_clique(image, new_w, smoothness_const, neighborhood_size, x, y, True)

            if new_energy < current_energy:
                if w[x - neighborhood_size, y - neighborhood_size] == 1:
                    w[x - neighborhood_size, y - neighborhood_size] = 0
                else:
                    w[x - neighborhood_size, y - neighborhood_size] = 1

            # if np.abs(new_energy - current_energy) < 1:
            #     return w
    return w


def iterated_conditional_modes_interlaced(image, w1, contour, neighborhood_size, smoothness_const, alpha):
    w = copy.copy(w1)
    dims = w.shape
    new_w = np.array(np.pad(w, neighborhood_size, 'edge'), dtype=np.int32)
    image = np.array(np.pad(image, neighborhood_size, 'edge'), dtype=np.int32)
    contour_matrix = np.zeros(dims, dtype=np.int32)
    contour_matrix = np.pad(cv2.drawContours(contour_matrix, contour, -1, 1, -1), neighborhood_size, 'edge')

    expected_val_in = np.bincount(new_w[contour_matrix == 1]).argmax()
    expected_val_out = np.logical_not(expected_val_in)

    for x in range(neighborhood_size, dims[0] + neighborhood_size):
        for y in range(neighborhood_size, dims[1] + neighborhood_size):
            current_energy = region_segmentation_cost_clique_interlaced(image, new_w, smoothness_const, contour_matrix,
                                                                        alpha, neighborhood_size, expected_val_in,
                                                                        expected_val_out, x, y)

            new_energy = region_segmentation_cost_clique_interlaced(image, new_w, smoothness_const, contour_matrix,
                                                                    alpha, neighborhood_size, expected_val_in,
                                                                    expected_val_out, x, y, True)

            if new_energy < current_energy:
                if w[x - neighborhood_size, y - neighborhood_size] == 1:
                    w[x - neighborhood_size, y - neighborhood_size] = 0
                else:
                    w[x - neighborhood_size, y - neighborhood_size] = 1

    return w


def boundary_finding(coefficients):
    acc = 0.01
    img_gradient_fn = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
    return boundary_segmentation_cost(img_gradient_fn, fourier_parametrization_to_indices(coefficients, acc))


img = cv2.imread('IMD003.bmp', 0)

gaussian_noise = np.random.normal(0, 70, size=(img.shape[0], img.shape[1]))

img_noise_temp = np.array(img, dtype=np.int32) + gaussian_noise
img_noise_temp[img_noise_temp > 255] = 255
img_noise_temp[img_noise_temp < 0] = 0
img_noise = np.array(img_noise_temp, dtype=np.uint8)

img_gradient = cv2.Laplacian(img_noise, cv2.CV_64F, ksize=5)

et, img_cn = cv2.threshold(cv2.imread('IMD003_cn.bmp', 0), 125, 1, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(img_cn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boundary_cost = boundary_segmentation_cost(img_gradient, contours)

et2, img_cn2 = cv2.threshold(img, 150, 1, cv2.THRESH_BINARY)
contours2, hierarchy2 = cv2.findContours(img_cn2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boundary_cost2 = boundary_segmentation_cost(img_gradient, contours2)

img_contour = copy.copy(img_noise)
cv2.drawContours(img_contour, contours, -1, 255, 3)

img_contour2 = copy.copy(img_noise)
cv2.drawContours(img_contour2, contours2, -1, 255, 3)


init_fourier_coeffs = elliptic_fourier_descriptors(np.squeeze(contours[0]), order=1)[0]
init_fourier_coeffs2 = np.append([240, 160], init_fourier_coeffs)

training_gradient_fun = grad(boundary_finding)
for iteration in range(10):
    init_fourier_coeffs2 -= training_gradient_fun(init_fourier_coeffs2) * 0.1

contour_from_fourier = fourier_parametrization_to_indices(init_fourier_coeffs2, 0.01)
img_contour3 = copy.copy(img_noise)
cv2.drawContours(img_contour3, contour_from_fourier, -1, 255, 3)


init_tr = 180
clique_size = 1
sm_const = 20
scaling_const = 800
max_iterations = 3
ret, best_img_seg = cv2.threshold(img_noise, init_tr, 1, cv2.THRESH_BINARY_INV)

start_time = time.time()

gd_segmentation = iterated_conditional_modes(img_noise, best_img_seg, clique_size, sm_const)
gd_segmentation2 = iterated_conditional_modes_interlaced(img_noise, best_img_seg, contours, clique_size, sm_const,
                                                         scaling_const)
for iteration2 in range(max_iterations):
    gd_segmentation = iterated_conditional_modes(img_noise, gd_segmentation, clique_size, sm_const)
    gd_segmentation2 = iterated_conditional_modes_interlaced(img_noise, gd_segmentation2, contours, clique_size,
                                                             sm_const, scaling_const)
final_time = time.time() - start_time

fig, ax = plt.subplots(1, 5)
plt.setp(ax, xticks=[], yticks=[])
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(img_noise, cmap='gray')
ax[1].set_title('Noisy')
ax[2].imshow(best_img_seg, cmap='gray')
ax[2].set_title(f'Initial segmentation (threshold {init_tr})')
ax[3].imshow(gd_segmentation, cmap='gray')
ax[3].set_title(f'ICM ({max_iterations} iterations)')
ax[4].imshow(gd_segmentation2, cmap='gray')
ax[4].set_title(f'ICM interlaced ({max_iterations} iterations, time: {final_time:.2f}s)')

fig2, ax2 = plt.subplots(1, 5)
plt.setp(ax2, xticks=[], yticks=[])
ax2[0].imshow(img_noise, cmap='gray')
ax2[0].set_title('Original')
ax2[1].imshow(img_gradient, cmap='gray')
ax2[1].set_title('Gradient')
ax2[2].imshow(img_contour, cmap='gray')
ax2[2].set_title(f'Contour (cost {boundary_cost})')
ax2[3].imshow(img_contour2, cmap='gray')
ax2[3].set_title(f'Contour (cost {boundary_cost2})')
ax2[4].imshow(img_contour3, cmap='gray')
ax2[4].set_title('Contour from Fourier parametrization')

plt.show()

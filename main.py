import cv2
import numpy as np
from scipy import optimize
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


def boundary_segmentation_cost_interlaced(image, contour, segmentation, beta):
    global full_b_cost
    b_cost = 0
    dims = segmentation.shape

    for i in range(len(contour[0])):
        x = contour[0][i][0][0]
        y = contour[0][i][0][1]
        b_cost += image[x, y]

    full_b_cost += b_cost

    contour_matrix = np.zeros(dims, dtype=np.int32)
    contour_matrix = cv2.drawContours(contour_matrix, contour, -1, 1, -1)

    region_module_influence = np.sum(segmentation[contour_matrix == 1])

    return full_b_cost + (beta * region_module_influence)


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


def icm_interlaced_wrapped(contour_coeffs):
    global region_segmentation2

    current_contour = fourier_parametrization_to_indices(contour_coeffs, 0.01)

    region_segmentation2 = iterated_conditional_modes_interlaced(img_noise, region_segmentation2, current_contour,
                                                                 clique_size, sm_const, scaling_const)


def boundary_finding(coefficients, *args):
    return -boundary_segmentation_cost(args[0], fourier_parametrization_to_indices(coefficients, args[1]))


def boundary_finding_interlaced(coefficients, *args):
    return -boundary_segmentation_cost_interlaced(args[0], fourier_parametrization_to_indices(coefficients, args[1]),
                                                  args[2], args[3])


img = cv2.imread('test_heart.png', 0)

gaussian_noise = np.random.normal(0, 50, size=(img.shape[0], img.shape[1]))

img_noise_temp = np.array(img, dtype=np.int32) + gaussian_noise
img_noise_temp[img_noise_temp > 255] = 255
img_noise_temp[img_noise_temp < 0] = 0
img_noise = np.array(img_noise_temp, dtype=np.uint8)

img_gradient = cv2.Laplacian(img_noise, cv2.CV_64F, ksize=1)
img_gradient = np.array(np.absolute(img_gradient), dtype=np.uint32)

et, img_cn = cv2.threshold(cv2.imread('contour5.png', 0), 125, 1, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(img_cn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boundary_cost = boundary_segmentation_cost(img_gradient, contours)

et2, img_cn2 = cv2.threshold(img, 150, 1, cv2.THRESH_BINARY)
contours2, hierarchy2 = cv2.findContours(img_cn2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boundary_cost2 = boundary_segmentation_cost(img_gradient, contours2)

img_contour = copy.copy(img_noise)
cv2.drawContours(img_contour, contours, -1, 255, 1)

img_contour2 = copy.copy(img_noise)
cv2.drawContours(img_contour2, contours2, -1, 255, 1)

init_tr = 180
clique_size = 1
sm_const = 15
scaling_const = 800
max_iterations = 10
ret, init_img_seg = cv2.threshold(img_noise, init_tr, 1, cv2.THRESH_BINARY)

start_time_region = time.time()

region_segmentation = iterated_conditional_modes(img_noise, init_img_seg, clique_size, sm_const)
region_segmentation2 = iterated_conditional_modes_interlaced(img_noise, init_img_seg, contours, clique_size, sm_const,
                                                             scaling_const)

final_time_region = time.time() - start_time_region

M = cv2.moments(img_cn)

init_fourier_coeffs = elliptic_fourier_descriptors(np.squeeze(contours[0]), order=4).flatten()
init_fourier_coeffs2 = np.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])], init_fourier_coeffs)

full_b_cost = 0

start_time_boundary = time.time()

optimized_fourier_coeffs = optimize.minimize(boundary_finding_interlaced, x0=init_fourier_coeffs2,
                                             args=(img_gradient, 0.01, region_segmentation2, scaling_const),
                                             method='Nelder-Mead', options={'maxiter': max_iterations, 'disp': False},
                                             tol=1, callback=icm_interlaced_wrapped).x

final_time_boundary = time.time() - start_time_boundary

optimized_contour = fourier_parametrization_to_indices(optimized_fourier_coeffs, 0.01)
img_contour_optimized = copy.copy(img_noise)
cv2.drawContours(img_contour_optimized, optimized_contour, -1, 255, 1)

fig, ax = plt.subplots(1, 5)
plt.setp(ax, xticks=[], yticks=[])
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(img_noise, cmap='gray')
ax[1].set_title('Noisy')
ax[2].imshow(init_img_seg, cmap='gray')
ax[2].set_title(f'Initial segmentation (threshold {init_tr})')
ax[3].imshow(region_segmentation, cmap='gray')
ax[3].set_title(f'ICM ({max_iterations} iterations)')
ax[4].imshow(region_segmentation2, cmap='gray')
ax[4].set_title(f'ICM interlaced ({max_iterations} iterations, time: {final_time_region:.2f}s)')

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
ax2[4].imshow(img_contour_optimized, cmap='gray')
ax2[4].set_title(f'Optimized contour (time: {final_time_boundary:.2f}s)')

plt.show()

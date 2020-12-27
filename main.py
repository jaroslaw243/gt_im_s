import cv2
import numpy as np
from scipy import optimize
from pyefd import elliptic_fourier_descriptors, calculate_dc_coefficients, reconstruct_contour
from matplotlib import pyplot as plt
import matplotlib
import copy
import time


def fourier_parametrization_to_indices(a0_c0, coefficients, accuracy):
    k_max = int(coefficients.size / 4)

    contour_array = []
    for t in np.arange(0, 2 * np.pi, accuracy):
        second_term = np.zeros((1, 2), dtype=np.float)
        coef_indices = np.array((0, 1, 3, 4), dtype=np.uint16)

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

        contour_array.append(indices)

    return [np.array(contour_array, dtype=np.int32)]


def reconstructed_contour_to_opencv_contour(contour_array):
    opencv_contour_array = []
    for ind in range(contour_array.shape[0]):
        opencv_contour_array.append([np.rint([contour_array[ind, 0], contour_array[ind, 1]])])

    return [np.array(opencv_contour_array, dtype=np.int32)]


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

    for i in range(len(contour[0])):
        x = contour[0][i][0][0]
        y = contour[0][i][0][1]
        b_cost += image[x, y]

    full_b_cost += b_cost

    contour_matrix = np.zeros(segmentation.shape, dtype=np.int32)
    contour_matrix = cv2.drawContours(contour_matrix, contour, -1, 1, -1)

    region_module_influence = np.sum(segmentation[contour_matrix == 1])

    return b_cost + (beta * region_module_influence)


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

    # expected_val_in = np.bincount(new_w[contour_matrix == 1]).argmax()
    # expected_val_out = np.logical_not(expected_val_in)

    expected_val_in = 1
    expected_val_out = 0

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

    current_contour = reconstructed_contour_to_opencv_contour(
        reconstruct_contour(locus=init_fourier_coeffs_first_part, coeffs=np.reshape(contour_coeffs, (-1, 4)),
                            num_points=p2c_acc))

    region_segmentation2 = iterated_conditional_modes_interlaced(img_noise, region_segmentation2, current_contour,
                                                                 clique_size, sm_const, scaling_const_alpha)


def boundary_finding(coefficients, *args):
    return -boundary_segmentation_cost(args[0], reconstructed_contour_to_opencv_contour(
        reconstruct_contour(locus=args[1], coeffs=np.reshape(coefficients, (-1, 4)),
                            num_points=args[2])))


def boundary_finding_interlaced(coefficients, *args):
    current_contour = reconstructed_contour_to_opencv_contour(
        reconstruct_contour(locus=args[1], coeffs=np.reshape(coefficients, (-1, 4)),
                            num_points=args[2]))

    return -boundary_segmentation_cost_interlaced(args[0], current_contour, args[3], args[4])


img = cv2.imread('test_complex.png', 0)

gaussian_noise = np.random.normal(0, 150, size=(img.shape[0], img.shape[1]))

img_noise_temp = np.array(img, dtype=np.int32) + gaussian_noise
img_noise_temp[img_noise_temp > 255] = 255
img_noise_temp[img_noise_temp < 0] = 0
img_noise = np.array(img_noise_temp, dtype=np.uint8)

img_gradient = cv2.Laplacian(img_noise, cv2.CV_64F, ksize=5)
img_gradient = np.array(np.absolute(img_gradient), dtype=np.uint32)

et, img_cn = cv2.threshold(cv2.imread('contour_complex3.png', 0), 125, 1, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(img_cn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

M = cv2.moments(img_cn)

init_fourier_coeffs_first_part = np.array(calculate_dc_coefficients(np.squeeze(contours[0])), dtype=np.float)
init_fourier_coeffs_second_part = elliptic_fourier_descriptors(np.squeeze(contours[0]), order=18,
                                                               normalize=False)

img_contour = copy.copy(img)
cv2.drawContours(img_contour,
                 reconstructed_contour_to_opencv_contour(reconstruct_contour(locus=init_fourier_coeffs_first_part,
                                                                             coeffs=init_fourier_coeffs_second_part,
                                                                             num_points=10000)), -1, 0, 1)

init_tr = 180
clique_size = 1
sm_const = 15
scaling_const_alpha = 500
scaling_const_beta = 2500
max_iterations = 10
p2c_acc = 10000
ret, init_img_seg = cv2.threshold(img_noise, init_tr, 1, cv2.THRESH_BINARY)

start_time_region = time.time()

region_segmentation = iterated_conditional_modes(img_noise, init_img_seg, clique_size, sm_const)
region_segmentation2 = iterated_conditional_modes_interlaced(img_noise, init_img_seg, contours, clique_size, sm_const,
                                                             scaling_const_alpha)

final_time_region = time.time() - start_time_region

full_b_cost = boundary_segmentation_cost(img_gradient,
                                         reconstructed_contour_to_opencv_contour(
                                             reconstruct_contour(locus=init_fourier_coeffs_first_part,
                                                                 coeffs=init_fourier_coeffs_second_part,
                                                                 num_points=p2c_acc)))

boundary_cost = boundary_segmentation_cost_interlaced(img_gradient, contours, region_segmentation2, scaling_const_beta)

start_time_boundary = time.time()

optimized_fourier_coeffs = optimize.minimize(boundary_finding_interlaced, x0=init_fourier_coeffs_second_part,
                                             args=(img_gradient, init_fourier_coeffs_first_part, p2c_acc,
                                                   region_segmentation2, scaling_const_beta),
                                             method='Nelder-Mead', options={'maxiter': max_iterations, 'disp': False},
                                             callback=icm_interlaced_wrapped).x

final_time_boundary = time.time() - start_time_boundary

optimized_contour = reconstructed_contour_to_opencv_contour(
    reconstruct_contour(locus=init_fourier_coeffs_first_part, coeffs=np.reshape(optimized_fourier_coeffs, (-1, 4)),
                        num_points=p2c_acc))
img_contour_optimized = copy.copy(img)
cv2.drawContours(img_contour_optimized, optimized_contour, -1, 0, 1)
optimized_boundary_cost = boundary_segmentation_cost_interlaced(img_gradient, optimized_contour, region_segmentation2,
                                                                scaling_const_beta)

matplotlib.use('TkAgg')

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

fig2, ax2 = plt.subplots(1, 4)
plt.setp(ax2, xticks=[], yticks=[])
ax2[0].imshow(img_noise, cmap='gray')
ax2[0].set_title('Original')
ax2[1].imshow(img_gradient, cmap='gray')
ax2[1].set_title('Gradient')
ax2[2].imshow(img_contour, cmap='gray')
ax2[2].set_title(f'Initial contour (cost {boundary_cost})')
ax2[3].imshow(img_contour_optimized, cmap='gray')
ax2[3].set_title(f'Optimized contour (cost {optimized_boundary_cost}, time: {final_time_boundary:.2f}s)')

plt.show()

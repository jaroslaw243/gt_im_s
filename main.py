import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import copy
import time
from pyefd import reconstruct_contour
from scipy import optimize
from game_theoretic_framework import GameTheoreticFramework

img = cv2.imread('test_complex2.png', 0)

noise_mean = 0
noise_var = 50
gaussian_noise = np.random.normal(noise_mean, noise_var, size=img.shape)

img_noise_temp = np.array(img, dtype=np.int32) + gaussian_noise
img_noise_temp[img_noise_temp > 255] = 255
img_noise_temp[img_noise_temp < 0] = 0
img_noise = np.array(img_noise_temp, dtype=np.uint8)

et, img_cn = cv2.threshold(cv2.imread('contour_complex3.png', 0), 125, 1, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(img_cn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

gt_segmentation = GameTheoreticFramework(image=img_noise, init_tr=180, clique_size=2, sm_const=10,
                                         scaling_const_alpha=0.1, scaling_const_beta=0.1, max_iterations=10,
                                         p2c_acc=2000, order_of_fourier_coeffs=12, init_contours=contours,
                                         img_gradient_ksize=21)

img_contour = copy.copy(img)
cv2.drawContours(img_contour,
                 gt_segmentation.reconstructed_contour_to_opencv_contour(
                     reconstruct_contour(locus=gt_segmentation.init_fourier_coeffs_first_part,
                                         coeffs=gt_segmentation.init_fourier_coeffs_second_part,
                                         num_points=gt_segmentation.p2c_acc)), -1, 0, 1)

bounds_width = 0.25
bounds_middle = gt_segmentation.init_fourier_coeffs_second_part.flatten()
lb = bounds_middle - np.abs(bounds_middle * bounds_width)
ub = bounds_middle + np.abs(bounds_middle * bounds_width)
bounds_gt = optimize.Bounds(lb, ub)

start_time = time.time()

optimized_fourier_coeffs = optimize.differential_evolution(func=gt_segmentation.boundary_finding_interlaced,
                                                           bounds=bounds_gt, maxiter=gt_segmentation.max_iterations,
                                                           callback=gt_segmentation.icm_interlaced_wrapped,
                                                           x0=gt_segmentation.init_fourier_coeffs_second_part.flatten()).x

finish_time = time.time() - start_time

optimized_contour = gt_segmentation.reconstructed_contour_to_opencv_contour(
    reconstruct_contour(locus=gt_segmentation.init_fourier_coeffs_first_part,
                        coeffs=np.reshape(optimized_fourier_coeffs, (-1, 4)),
                        num_points=gt_segmentation.p2c_acc))
img_contour_optimized = copy.copy(img)
cv2.drawContours(img_contour_optimized, optimized_contour, -1, 0, 1)

matplotlib.use('TkAgg')

fig, ax = plt.subplots(2, 4)
fig.suptitle(r'$\alpha = %.1f, \beta = %.1f, \lambda = %.2f, k_{size} = %d$' % (gt_segmentation.scaling_const_alpha,
                                                                                gt_segmentation.scaling_const_beta,
                                                                                gt_segmentation.sm_const,
                                                                                gt_segmentation.img_gradient_ksize),
             fontsize=32)
plt.setp(ax, xticks=[], yticks=[])
ax[0, 0].imshow(img, cmap='gray')
ax[0, 0].set_title('Original')
ax[0, 1].imshow(img_noise, cmap='gray')
ax[0, 1].set_title(r'Noisy ($\mu = %d, \sigma = %d$)' % (noise_mean, noise_var))
ax[0, 2].imshow(gt_segmentation.init_img_seg, cmap='gray')
ax[0, 2].set_title(f'Initial segmentation (threshold {gt_segmentation.init_tr})')
ax[0, 3].imshow(gt_segmentation.region_segmentation, cmap='gray')
ax[0, 3].set_title(f'ICM interlaced ({gt_segmentation.max_iterations} iterations)')

ax[1, 0].axis('off')

ax[1, 1].imshow(gt_segmentation.image_gradient, cmap='gray')
ax[1, 1].set_title('Gradient')
ax[1, 2].imshow(img_contour, cmap='gray')
ax[1, 2].set_title(f'Initial contour')
ax[1, 3].imshow(img_contour_optimized, cmap='gray')
ax[1, 3].set_title(
    f'Optimized contour (cost {(gt_segmentation.b_cost_interlaced / gt_segmentation.init_b_cost_interlaced):.2f},'
    f' time: {finish_time:.2f}s)')

plt.show()

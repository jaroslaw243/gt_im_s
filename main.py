import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import copy
import time
from pyefd import elliptic_fourier_descriptors, calculate_dc_coefficients, reconstruct_contour
from scipy import optimize


class GameTheoreticFramework:
    def __init__(self, image, init_tr, clique_size, sm_const, scaling_const_alpha, scaling_const_beta, max_iterations,
                 p2c_acc, order_of_fourier_coeffs, init_contours, object_brighter_than_background=True):
        self.image = image

        # for region based module
        self.init_tr = init_tr
        self.clique_size = clique_size
        self.sm_const = sm_const
        self.scaling_const_alpha = scaling_const_alpha
        self.scaling_const_beta = scaling_const_beta
        self.max_iterations = max_iterations
        self.object_brighter_than_background = object_brighter_than_background
        self.region_segmentation = None
        self.init_img_seg = None

        # for boundary finding module
        self.p2c_acc = p2c_acc
        self.order_of_fourier_coeffs = order_of_fourier_coeffs
        self.contours = init_contours
        self.prior_b_cost = 0
        self.image_gradient = None
        self.init_b_cost_interlaced = None
        self.b_cost_interlaced = None
        self.init_fourier_coeffs_first_part = None
        self.init_fourier_coeffs_second_part = None

        self.init_region_segmentation()
        self.init_boundary_finding()

    @staticmethod
    def reconstructed_contour_to_opencv_contour(contour_array):
        opencv_contour_array = []
        for ind in range(contour_array.shape[0]):
            opencv_contour_array.append([np.rint([contour_array[ind, 0], contour_array[ind, 1]])])

        return [np.array(opencv_contour_array, dtype=np.int32)]

    def init_region_segmentation(self):
        # binary thresholding is conducted to initialize region segmentation module,
        # initial labels are saved for comparison with final result
        if self.object_brighter_than_background:
            ret, self.init_img_seg = cv2.threshold(self.image, self.init_tr, 1, cv2.THRESH_BINARY)
        else:
            ret, self.init_img_seg = cv2.threshold(self.image, self.init_tr, 1, cv2.THRESH_BINARY_INV)

        self.region_segmentation = copy.copy(self.init_img_seg)

    def init_boundary_finding(self):
        # a0 and c0 are calculated using pyefd package
        self.init_fourier_coeffs_first_part = np.array(calculate_dc_coefficients(np.squeeze(self.contours[0])),
                                                       dtype=np.float)
        # an, bn, cn and dn are calculated using the same package
        self.init_fourier_coeffs_second_part = elliptic_fourier_descriptors(np.squeeze(self.contours[0]),
                                                                            order=self.order_of_fourier_coeffs,
                                                                            normalize=False)

        # gradient of the image is obtained
        img_gradient_neg = cv2.Laplacian(self.image, cv2.CV_64F, ksize=11)
        self.image_gradient = np.array(np.absolute(img_gradient_neg), dtype=np.uint32)

        # contour is remade using calculated fourier coefficients
        self.contours = self.reconstructed_contour_to_opencv_contour(
            reconstruct_contour(locus=self.init_fourier_coeffs_first_part,
                                coeffs=self.init_fourier_coeffs_second_part,
                                num_points=self.p2c_acc))

        # initial boundary cost is calculated and saved for comparison with final result
        self.init_b_cost_interlaced = self.boundary_segmentation_cost_interlaced()
        self.prior_b_cost = copy.copy(self.init_b_cost_interlaced)

    def boundary_segmentation_cost_interlaced(self):
        # contour is drawn on matrix of zeros with value of 1 and thickness 1,
        # we can use that matrix to obtain values that correspond to ones in gradient image
        drawn_contour = np.zeros(self.image_gradient.shape, dtype=np.int32)
        cv2.drawContours(drawn_contour, self.contours, -1, 1, 1)
        b_cost = np.sum(self.image_gradient[drawn_contour == 1])

        prior_b_cost_temp = copy.copy(self.prior_b_cost)
        self.prior_b_cost = b_cost

        # contour is drawn on matrix of zeros with value of 1 and thickness -1 (so filled)
        contour_matrix = np.zeros(self.region_segmentation.shape, dtype=np.int32)
        cv2.drawContours(contour_matrix, self.contours, -1, 1, -1)

        # it is assumed that background pixels are labeled with zeros,
        # they are replaced with a negative number so they decrease overall energy in boundary finding module
        image_r = copy.copy(self.region_segmentation)
        image_r = np.array(image_r, dtype=np.int8)
        image_r[image_r == 0] = -1

        region_module_influence = np.sum(image_r[contour_matrix == 1])

        return prior_b_cost_temp + b_cost + (self.scaling_const_beta * region_module_influence)

    def boundary_finding_interlaced(self, coefficients):
        self.contours = self.reconstructed_contour_to_opencv_contour(
            reconstruct_contour(locus=self.init_fourier_coeffs_first_part, coeffs=np.reshape(coefficients, (-1, 4)),
                                num_points=self.p2c_acc))
        self.b_cost_interlaced = self.boundary_segmentation_cost_interlaced()

        # return statement is for scipy's optimize function
        return -self.b_cost_interlaced

    def iterated_conditional_modes_interlaced(self):
        starting_region_segmentation = copy.copy(self.region_segmentation)
        dims = starting_region_segmentation.shape

        # image and label matrix are padded to avoid indexing errors
        new_w = np.array(np.pad(starting_region_segmentation, self.clique_size, 'edge'), dtype=np.int32)
        image_p = np.array(np.pad(self.image, self.clique_size, 'edge'), dtype=np.int32)
        contour_matrix = np.zeros(dims, dtype=np.int32)
        contour_matrix = np.pad(cv2.drawContours(contour_matrix, self.contours, -1, 1, -1), self.clique_size, 'edge')

        # it is assumed that object pixels are labeled as ones,
        # background as zeros (and that there are only those two possibilities)
        expected_val_in = 1
        expected_val_out = 0

        for x in range(self.clique_size, dims[0] + self.clique_size):
            for y in range(self.clique_size, dims[1] + self.clique_size):
                current_energy = self.region_segmentation_cost_clique_interlaced(image_p, new_w, contour_matrix,
                                                                                 expected_val_in,
                                                                                 expected_val_out, x, y)

                new_energy = self.region_segmentation_cost_clique_interlaced(image_p, new_w, contour_matrix,
                                                                             expected_val_in,
                                                                             expected_val_out, x, y, True)

                if new_energy < current_energy:
                    if starting_region_segmentation[x - self.clique_size, y - self.clique_size] == 1:
                        starting_region_segmentation[x - self.clique_size, y - self.clique_size] = 0
                    else:
                        starting_region_segmentation[x - self.clique_size, y - self.clique_size] = 1

        self.region_segmentation = starting_region_segmentation

    def region_segmentation_cost_clique_interlaced(self, padded_image, padded_segmentation, contour_m, u, v, i, j,
                                                   change=False):
        # clique is copied to new variable because we want to flip it's values to see if energy decreases,
        # without altering original matrix
        segmentation_clique = copy.copy(padded_segmentation[i - self.clique_size:i + (self.clique_size + 1),
                                        j - self.clique_size:j + (self.clique_size + 1)])
        image_clique = padded_image[i - self.clique_size:i + (self.clique_size + 1),
                                    j - self.clique_size:j + (self.clique_size + 1)]
        if change:
            if segmentation_clique[self.clique_size, self.clique_size] == 1:
                segmentation_clique[self.clique_size, self.clique_size] = 0
            else:
                segmentation_clique[self.clique_size, self.clique_size] = 1

        data_fidelity_term = np.sum(np.square(image_clique - segmentation_clique))
        smoothness_term = np.sum(
            np.square(segmentation_clique - segmentation_clique[self.clique_size, self.clique_size]))

        boundary_seg = contour_m[i - self.clique_size:i + (self.clique_size + 1),
                                 j - self.clique_size:j + (self.clique_size + 1)]

        sum_in = np.sum(np.square(segmentation_clique[boundary_seg == 1] - u))
        sum_out = np.sum(np.square(segmentation_clique[boundary_seg == 0] - v))

        return (data_fidelity_term + (self.sm_const ** 2) * smoothness_term) + (
                self.scaling_const_alpha * (sum_in + sum_out))

    def icm_interlaced_wrapped(self, contour_coeffs):
        self.iterated_conditional_modes_interlaced()


img = cv2.imread('test_complex3.png', 0)

noise_mean = 0
noise_var = 50
gaussian_noise = np.random.normal(noise_mean, noise_var, size=img.shape)

img_noise_temp = np.array(img, dtype=np.int32) + gaussian_noise
img_noise_temp[img_noise_temp > 255] = 255
img_noise_temp[img_noise_temp < 0] = 0
img_noise = np.array(img_noise_temp, dtype=np.uint8)

et, img_cn = cv2.threshold(cv2.imread('contour_complex2.png', 0), 125, 1, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(img_cn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

gt_segmentation = GameTheoreticFramework(image=img_noise, init_tr=180, clique_size=1, sm_const=8,
                                         scaling_const_alpha=300, scaling_const_beta=1.0, max_iterations=10,
                                         p2c_acc=2000, order_of_fourier_coeffs=12, init_contours=contours)

img_contour = copy.copy(img)
cv2.drawContours(img_contour,
                 gt_segmentation.reconstructed_contour_to_opencv_contour(
                     reconstruct_contour(locus=gt_segmentation.init_fourier_coeffs_first_part,
                                         coeffs=gt_segmentation.init_fourier_coeffs_second_part,
                                         num_points=gt_segmentation.p2c_acc)), -1, 0, 1)

start_time_boundary = time.time()

optimized_fourier_coeffs = optimize.minimize(gt_segmentation.boundary_finding_interlaced,
                                             x0=gt_segmentation.init_fourier_coeffs_second_part,
                                             method='Nelder-Mead',
                                             options={'maxiter': gt_segmentation.max_iterations, 'disp': False},
                                             callback=gt_segmentation.icm_interlaced_wrapped).x

final_time_boundary = time.time() - start_time_boundary

optimized_contour = gt_segmentation.reconstructed_contour_to_opencv_contour(
    reconstruct_contour(locus=gt_segmentation.init_fourier_coeffs_first_part,
                        coeffs=np.reshape(optimized_fourier_coeffs, (-1, 4)),
                        num_points=gt_segmentation.p2c_acc))
img_contour_optimized = copy.copy(img)
cv2.drawContours(img_contour_optimized, optimized_contour, -1, 0, 1)
optimized_boundary_cost = 1

matplotlib.use('TkAgg')

fig, ax = plt.subplots(1, 4)
plt.setp(ax, xticks=[], yticks=[])
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(img_noise, cmap='gray')
ax[1].set_title(r'Noisy ($\mu = %d, \sigma = %d$)' % (noise_mean, noise_var))
ax[2].imshow(gt_segmentation.init_img_seg, cmap='gray')
ax[2].set_title(f'Initial segmentation (threshold {gt_segmentation.init_tr})')
ax[3].imshow(gt_segmentation.region_segmentation, cmap='gray')
ax[3].set_title(f'ICM interlaced ({gt_segmentation.max_iterations} iterations)')

fig2, ax2 = plt.subplots(1, 4)
plt.setp(ax2, xticks=[], yticks=[])
ax2[0].imshow(img_noise, cmap='gray')
ax2[0].set_title('Image')
ax2[1].imshow(gt_segmentation.image_gradient, cmap='gray')
ax2[1].set_title('Gradient')
ax2[2].imshow(img_contour, cmap='gray')
ax2[2].set_title(f'Initial contour')
ax2[3].imshow(img_contour_optimized, cmap='gray')
ax2[3].set_title(
    f'Optimized contour (cost {(gt_segmentation.b_cost_interlaced / gt_segmentation.init_b_cost_interlaced):.2f},'
    f' time: {final_time_boundary:.2f}s)')

plt.show()

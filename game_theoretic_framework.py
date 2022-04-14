import cv2
import numpy as np
import copy
from pyefd import elliptic_fourier_descriptors, calculate_dc_coefficients, reconstruct_contour


class GameTheoreticFramework:
    def __init__(self, image_path, init_tr, clique_size, sm_const, scaling_const_alpha, scaling_const_beta,
                 max_iterations, p2c_acc, order_of_fourier_coeffs, init_contours, img_gradient_ksize,
                 region_seg_expected_vals_in_and_out=(1, 0), object_brighter_than_background=True, full_init=True):
        self.max_iterations = max_iterations
        self.image_path = image_path
        self.image = None
        self.iter_num = 0

        # for region based module
        self.scaling_const_alpha = scaling_const_alpha
        self.init_tr = init_tr
        self.clique_size = clique_size
        self.sm_const = sm_const
        self.expected_val_in = region_seg_expected_vals_in_and_out[0]
        self.expected_val_out = region_seg_expected_vals_in_and_out[1]
        self.object_brighter_than_background = object_brighter_than_background
        self.region_segmentation = None
        self.init_img_seg = None

        # for boundary finding module
        self.scaling_const_beta = scaling_const_beta
        self.p2c_acc = p2c_acc
        self.order_of_fourier_coeffs = order_of_fourier_coeffs
        self.img_gradient_ksize = img_gradient_ksize
        self.init_contours = init_contours
        self.contours = None
        self.prior_b_cost = 0
        self.image_gradient = None
        self.init_b_cost_interlaced = None
        self.b_cost_interlaced = None
        self.init_fourier_coeffs_first_part = None
        self.init_fourier_coeffs_second_part = None

        if full_init:
            self.load_image()
            self.run_full_init()

    @staticmethod
    def reconstructed_contour_to_opencv_contour(contour_array):
        opencv_contour_array = []
        for ind in range(contour_array.shape[0]):
            opencv_contour_array.append([np.rint([contour_array[ind, 0], contour_array[ind, 1]])])

        return [np.array(opencv_contour_array, dtype=np.int32)]

    def load_image(self):
        self.image = cv2.imread(self.image_path, 0)

    def run_full_init(self):
        self.init_region_segmentation()
        self.init_boundary_finding()

    def init_region_segmentation(self):
        # binary thresholding is conducted to initialize region segmentation module,
        # initial labels are saved for comparison with final result
        if self.object_brighter_than_background:
            ret, self.init_img_seg = cv2.threshold(self.image, self.init_tr, 1, cv2.THRESH_BINARY)
        else:
            ret, self.init_img_seg = cv2.threshold(self.image, self.init_tr, 1, cv2.THRESH_BINARY_INV)

        self.region_segmentation = copy.copy(self.init_img_seg)

    def init_boundary_finding(self):
        if isinstance(self.init_contours, str):
            et, img_cn = cv2.threshold(cv2.imread(self.init_contours, 0), 125, 1, cv2.THRESH_BINARY)
            self.contours, hierarchy = cv2.findContours(img_cn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        elif isinstance(self.init_contours, list):
            self.contours = self.init_contours

        else:
            raise TypeError("'init_contours' can only be one of these two:\n"
                            "- path to a image that after binarization, with threshold equal to 125, will be suitable"
                            " for openCV function 'findContours';\n"
                            "- python list containing openCV contour.\n")

        # a0 and c0 are calculated using pyefd package
        self.init_fourier_coeffs_first_part = np.array(calculate_dc_coefficients(np.squeeze(self.contours[0])),
                                                       dtype=np.float)
        # an, bn, cn and dn are calculated using the same package
        self.init_fourier_coeffs_second_part = elliptic_fourier_descriptors(np.squeeze(self.contours[0]),
                                                                            order=self.order_of_fourier_coeffs,
                                                                            normalize=False)

        # gradient of the image is obtained
        img_gradient_neg = cv2.Laplacian(self.image, cv2.CV_64F, ksize=self.img_gradient_ksize)
        self.image_gradient = np.absolute(img_gradient_neg)
        self.image_gradient = np.array((self.image_gradient / np.amax(self.image_gradient)) * 255, dtype=np.uint8)

        # contour is remade using calculated fourier coefficients
        self.contours = self.reconstructed_contour_to_opencv_contour(
            reconstruct_contour(locus=self.init_fourier_coeffs_first_part,
                                coeffs=self.init_fourier_coeffs_second_part,
                                num_points=self.p2c_acc))

        # initial boundary cost is calculated and saved for comparison with final result
        self.init_b_cost_interlaced = self.boundary_segmentation_cost_interlaced()
        self.prior_b_cost = copy.copy(self.init_b_cost_interlaced)

    def boundary_segmentation_cost_interlaced(self):
        # contour is drawn on matrix of zeros, with value of 1 and thickness 1,
        # we can use that matrix to obtain values that correspond to 1 in gradient image (so they lay on the contour)
        drawn_contour = np.zeros(self.image_gradient.shape, dtype=np.int32)
        cv2.drawContours(drawn_contour, self.contours, -1, 1, 1)
        b_cost = np.sum(self.image_gradient[drawn_contour == 1])

        prior_b_cost_temp = copy.copy(self.prior_b_cost)
        self.prior_b_cost = b_cost

        # contour is drawn on matrix of zeros with value of 1 and thickness -1 (so filled)
        contour_matrix = np.zeros(self.region_segmentation.shape, dtype=np.int32)
        cv2.drawContours(contour_matrix, self.contours, -1, 1, -1)

        image_r = copy.copy(self.region_segmentation)
        image_r = np.array(image_r, dtype=np.int8)

        # background pixel labels are replaced with a negative number,
        # so they decrease overall energy in boundary finding module
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
        # contour is drawn on matrix of zeros with value of 1 and thickness -1 (so filled)
        contour_matrix = np.zeros(dims, dtype=np.int32)
        contour_matrix = np.pad(cv2.drawContours(contour_matrix, self.contours, -1, 1, -1), self.clique_size, 'edge')

        for x in range(self.clique_size, dims[0] + self.clique_size):
            for y in range(self.clique_size, dims[1] + self.clique_size):
                current_energy = self.region_segmentation_cost_clique_interlaced(image_p, new_w, contour_matrix, x, y)

                new_energy = self.region_segmentation_cost_clique_interlaced(image_p, new_w, contour_matrix, x, y, True)

                if new_energy < current_energy:
                    if starting_region_segmentation[x - self.clique_size, y - self.clique_size] == 1:
                        starting_region_segmentation[x - self.clique_size, y - self.clique_size] = 0
                    else:
                        starting_region_segmentation[x - self.clique_size, y - self.clique_size] = 1

        self.region_segmentation = starting_region_segmentation

    def region_segmentation_cost_clique_interlaced(self, padded_image, padded_segmentation, contour_m, i, j,
                                                   change=False):
        # clique is copied to new variable because we want to flip its values to see if energy decreases,
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

        sum_in = np.sum(np.square(segmentation_clique[boundary_seg == 1] - self.expected_val_in))
        sum_out = np.sum(np.square(segmentation_clique[boundary_seg == 0] - self.expected_val_out))

        return (data_fidelity_term + (self.sm_const ** 2) * smoothness_term) + (
                self.scaling_const_alpha * (sum_in + sum_out))

    def icm_interlaced_wrapped(self, contour_coeffs, convergence):
        self.iterated_conditional_modes_interlaced()

        self.iter_num += 1
        print('Iteration nr ', self.iter_num)
        return False

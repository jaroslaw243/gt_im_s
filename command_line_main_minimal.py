import cv2
import numpy as np
from pyefd import reconstruct_contour
from scipy import optimize
import yaml
import argparse
import os
from game_theoretic_framework import GameTheoreticFramework


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="yaml file with GameTheoreticFramework object", required=True)
parser.add_argument("-od", "--output_directory", help="path to output directory (with slash at the end)",
                    required=False, default='')
args = parser.parse_args()

with open(args.input) as file:
    gt_segmentation = yaml.load(file, Loader=yaml.Loader)

gt_segmentation.load_image()
gt_segmentation.run_full_init()

bounds_width = 0.55
bounds_middle = gt_segmentation.init_fourier_coeffs_second_part.flatten()
lb = bounds_middle - np.abs(bounds_middle * bounds_width)
ub = bounds_middle + np.abs(bounds_middle * bounds_width)
bounds_gt = optimize.Bounds(lb, ub)

optimized_fourier_coeffs = optimize.differential_evolution(func=gt_segmentation.boundary_finding_interlaced,
                                                           bounds=bounds_gt, maxiter=gt_segmentation.max_iterations,
                                                           callback=gt_segmentation.icm_interlaced_wrapped).x

optimized_contour = gt_segmentation.reconstructed_contour_to_opencv_contour(
    reconstruct_contour(locus=gt_segmentation.init_fourier_coeffs_first_part,
                        coeffs=np.reshape(optimized_fourier_coeffs, (-1, 4)),
                        num_points=gt_segmentation.p2c_acc))

optimized_contour_matrix = np.zeros(gt_segmentation.region_segmentation.shape,
                                    dtype=gt_segmentation.region_segmentation.dtype)
cv2.drawContours(optimized_contour_matrix, optimized_contour, -1, 1, -1)

image_name = os.path.basename(gt_segmentation.image_path)
image_name = os.path.splitext(image_name)[0]

cv2.imwrite(args.output_directory + image_name + '_region' + '.png',
            gt_segmentation.region_segmentation, params=(cv2.IMWRITE_PNG_BILEVEL, 1))
cv2.imwrite(args.output_directory + image_name + '_contour' + '.png',
            optimized_contour_matrix, params=(cv2.IMWRITE_PNG_BILEVEL, 1))

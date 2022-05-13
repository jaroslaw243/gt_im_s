import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy
import time
from pyefd import reconstruct_contour
from scipy import optimize
import yaml
import argparse
from game_theoretic_framework import GameTheoreticFramework
from utils import dice

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="yaml file with GameTheoreticFramework object", required=True)
parser.add_argument("-r", "--reference", help="reference image for drawing contours and calculating Dice coefficient",
                    required=True)
parser.add_argument("-dk", "--dict_key", help="key used for output dict",
                    required=True)
parser.add_argument("-fn", "--figures_name", help="name used for output image with figures",
                    required=True)
parser.add_argument("-yn", "--yaml_name", help="name used for output yaml file",
                    required=True)
args = parser.parse_args()

ref_img = cv2.imread(args.reference, 0)

with open(args.input) as file:
    gt_segmentation = yaml.load(file, Loader=yaml.Loader)

gt_segmentation.load_image()
gt_segmentation.run_full_init()

filename = gt_segmentation.image_path.split(".")
filename = filename[0]
filename = filename.split("_")
noise_mean = int(filename[-2])
noise_sd = int(filename[-1])

img_contour = copy.copy(ref_img)
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
                                                           callback=gt_segmentation.icm_interlaced_wrapped).x

finish_time = time.time() - start_time

optimized_contour = gt_segmentation.reconstructed_contour_to_opencv_contour(
    reconstruct_contour(locus=gt_segmentation.init_fourier_coeffs_first_part,
                        coeffs=np.reshape(optimized_fourier_coeffs, (-1, 4)),
                        num_points=gt_segmentation.p2c_acc))
img_contour_optimized = copy.copy(ref_img)
cv2.drawContours(img_contour_optimized, optimized_contour, -1, 0, 1)

if gt_segmentation.object_brighter_than_background:
    ret, reference_seg = cv2.threshold(ref_img, gt_segmentation.init_tr, 1, cv2.THRESH_BINARY)
else:
    ret, reference_seg = cv2.threshold(ref_img, gt_segmentation.init_tr, 1, cv2.THRESH_BINARY_INV)

optimized_contour_matrix = np.zeros(reference_seg.shape, dtype=reference_seg.dtype)
cv2.drawContours(optimized_contour_matrix, optimized_contour, -1, 1, -1)

region_dice = dice(gt_segmentation.region_segmentation, reference_seg)
contour_dice = dice(optimized_contour_matrix, reference_seg)

dpi_for_fig = 150

fig, ax = plt.subplots(2, 4, figsize=(round(1920/dpi_for_fig), round(1080/dpi_for_fig)), dpi=dpi_for_fig)

fig.suptitle(
    r'$\alpha = %.1f, \beta = %.1f, \lambda = %.2f, %d \mathrm{ \ iterations}, \mathrm{time: \ }%.2f \mathrm{s}$' % (
        gt_segmentation.scaling_const_alpha, gt_segmentation.scaling_const_beta, gt_segmentation.sm_const,
        gt_segmentation.max_iterations, finish_time), fontsize=30)

plt.setp(ax, xticks=[], yticks=[])
ax[0, 0].imshow(ref_img, cmap='gray')
ax[0, 0].set_title(f'Original (height: {ref_img.shape[0]}px,\n width: {ref_img.shape[1]}px)')
ax[0, 1].imshow(gt_segmentation.image, cmap='gray')
ax[0, 1].set_title(r'Noisy ($\mu = %d, \sigma = %d$)' % (noise_mean, noise_sd))
ax[0, 2].imshow(gt_segmentation.init_img_seg, cmap='gray')
ax[0, 2].set_title(f'Initial segmentation\n (threshold {gt_segmentation.init_tr})')
ax[0, 3].imshow(gt_segmentation.region_segmentation, cmap='gray')
ax[0, 3].set_title(f'After ICM\n (Dice coefficient = {region_dice:.2f})')

ax[1, 0].axis('off')

ax[1, 1].imshow(gt_segmentation.image_gradient, cmap='gray')
ax[1, 1].set_title(r'Gradient ($k_{size} = %d$)' % (gt_segmentation.img_gradient_ksize))
ax[1, 2].imshow(img_contour, cmap='gray')
ax[1, 2].set_title(f'Initial contour')
ax[1, 3].imshow(img_contour_optimized, cmap='gray')
ax[1, 3].set_title(
    f'Optimized contour (cost change: {(gt_segmentation.b_cost_interlaced / gt_segmentation.init_b_cost_interlaced):.2f},'
    f'\n Dice coefficient = {contour_dice:.2f})')

plt.savefig(args.figures_name, bbox_inches='tight', dpi=dpi_for_fig)

output_dict = {'input_name': args.input, 'elapsed_time': finish_time,
               'region_dice': float(region_dice), 'contour_dice': float(contour_dice)}

with open(args.yaml_name, 'a') as file:
    yaml.safe_dump([{args.dict_key: output_dict}], file)

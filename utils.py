import copy
import numpy as np
import cv2
import os


def dice(calculated, reference, k=1):
    intersection = np.sum(calculated[reference == k]) * 2.0
    return intersection / (np.sum(calculated) + np.sum(reference))


def add_gaussian_noise(image, s_deviation, mean=0):
    if not (s_deviation == 0 and mean == 0):
        gaussian_noise = np.random.normal(mean, s_deviation, size=image.shape)

        img_noise_temp = np.array(image, dtype=np.int32) + gaussian_noise
        img_noise_temp[img_noise_temp > 255] = 255
        img_noise_temp[img_noise_temp < 0] = 0
        img_noise = np.array(img_noise_temp, dtype=np.uint8)
    else:
        img_noise = copy.copy(image)

    return img_noise


def hair_removal(image):
    org_image = copy.copy(image)

    # kernel with size 17x17 was working well on image with size 574x765,
    # so for other resolutions it is scaled according to given image width
    scale = image.shape[1] / 765
    kernel_size = round(17 * scale)
    if (kernel_size % 2) == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(1, (kernel_size, kernel_size))

    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    return cv2.inpaint(org_image, thresh2, 1, cv2.INPAINT_TELEA)


def image_list_ph2(beginning_of_the_path, first_image=1, last_image=25):
    return [beginning_of_the_path + r'\IMD%03d\IMD%03d_Dermoscopic_Image\IMD%03d.bmp' % (i, i, i) for i in
            range(first_image, last_image + 1)]


def fill_image_seg_boundaries(seg):
    org_seg = copy.copy(seg)
    inv_seg = cv2.bitwise_not(org_seg)

    filled_cor1 = cv2.floodFill(inv_seg, None, (0, 0), 1)
    filled_cor2 = cv2.floodFill(inv_seg, None, (inv_seg.shape[1] - 1, 0), 1)
    filled_cor3 = cv2.floodFill(inv_seg, None, (0, inv_seg.shape[0] - 1), 1)
    filled_cor4 = cv2.floodFill(inv_seg, None, (inv_seg.shape[1] - 1, inv_seg.shape[0] - 1), 1)

    comb1 = filled_cor1[1] & filled_cor2[1]
    comb2 = comb1 & filled_cor3[1]
    comb3 = comb2 & filled_cor4[1]

    org_seg[comb3 == 1] = 0

    return org_seg


def hair_removal_and_fill_image_seg_boundaries(image, seg):
    org_image = copy.copy(image)
    org_seg = copy.copy(seg)
    inv_seg = cv2.bitwise_not(org_seg)

    filled_cor1 = cv2.floodFill(inv_seg, None, (0, 0), 1)
    filled_cor2 = cv2.floodFill(inv_seg, None, (inv_seg.shape[1] - 1, 0), 1)
    filled_cor3 = cv2.floodFill(inv_seg, None, (0, inv_seg.shape[0] - 1), 1)
    filled_cor4 = cv2.floodFill(inv_seg, None, (inv_seg.shape[1] - 1, inv_seg.shape[0] - 1), 1)

    comb1 = filled_cor1[1] & filled_cor2[1]
    comb2 = comb1 & filled_cor3[1]
    comb3 = comb2 & filled_cor4[1]

    # kernel with size 17x17 was working well on image with size 574x765,
    # so for other resolutions it is scaled according to given image width
    scale = image.shape[1] / 765
    kernel_size = round(17 * scale)
    if (kernel_size % 2) == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(1, (kernel_size, kernel_size))

    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    combined_mask = thresh2
    combined_mask[comb3 == 1] = 255

    return cv2.inpaint(org_image, combined_mask, 1, cv2.INPAINT_TELEA)


def make_initial_contours(input_files, input_folder, output_folder):
    for file in input_files:
        org_contour = cv2.imread(input_folder + file, 0)
        mask_size1 = np.random.randint(20, 51, dtype=int)
        mask_size2 = np.random.randint(20, 31, dtype=int)
        mask_size3 = np.random.randint(20, 81, dtype=int)

        kernel1 = np.ones((mask_size1, mask_size1), np.uint8)
        kernel2 = np.ones((mask_size2, mask_size2), np.uint8)
        kernel3 = np.ones((mask_size3, mask_size3), np.uint8)

        contour_dilation1 = cv2.dilate(org_contour, kernel1, iterations=3)
        contour_erosion2 = cv2.erode(contour_dilation1, kernel2, iterations=2)
        contour_dilation3 = cv2.dilate(contour_erosion2, kernel3, iterations=1)

        cv2.imwrite(output_folder + os.path.basename(file), contour_dilation3, params=(cv2.IMWRITE_PNG_BILEVEL, 1))

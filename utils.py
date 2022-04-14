import copy
import numpy as np
import cv2


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
    kernel = cv2.getStructuringElement(1, (17, 17))

    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    return cv2.inpaint(org_image, thresh2, 1, cv2.INPAINT_TELEA)


def image_list_ph2(beginning_of_the_path, first_image=1, last_image=25):
    return [beginning_of_the_path + r'\IMD%03d\IMD%03d_Dermoscopic_Image\IMD%03d.bmp' % (i, i, i) for i in
            range(first_image, last_image + 1)]

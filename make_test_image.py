import cv2
from utils import add_gaussian_noise

input_filename = 'test_complex2'

img = cv2.imread(input_filename + '.png', 0)

noise_mean = 0
noise_sd = 100

img_noise = add_gaussian_noise(image=img, s_deviation=noise_sd, mean=noise_mean)

output_filename = f'{input_filename}_{noise_mean}_{noise_sd}.png'

cv2.imwrite(output_filename, img_noise)

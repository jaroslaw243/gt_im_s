import yaml
from game_theoretic_framework import GameTheoreticFramework

gt_segmentation = GameTheoreticFramework(image_path='test_complex2_0_100.png', init_tr=180, clique_size=4,
                                         sm_const=14, scaling_const_alpha=0.1, scaling_const_beta=0.1,
                                         max_iterations=10, p2c_acc=2000, order_of_fourier_coeffs=14,
                                         init_contours='contour_complex3.png', img_gradient_ksize=29, full_init=False)

with open(r'.\gt_segmentation.yaml', 'w') as file:
    documents = yaml.dump(gt_segmentation, file)

from matplotlib import pyplot as plt
import matplotlib
import yaml
from game_theoretic_framework import GameTheoreticFramework

with open(r'.\gt_segmentation_ph2.yaml') as file:
    gt_segmentation = yaml.load(file, Loader=yaml.Loader)

# running these two is necessary only if you don't have the array representing image saved in the loaded yaml file
# (so when 'full_init' was set to 'False' when creating that yaml file)
gt_segmentation.load_image()
gt_segmentation.run_full_init()

# for dermoscopic images 'contour' module output is generally better
contour_mask = gt_segmentation.run_segmentation(return_region=False)

matplotlib.use('TkAgg')

fig, ax = plt.subplots(1, 2)

fig.suptitle(
    r'$\alpha = %.1f, \beta = %.1f, \lambda = %.2f, %d \mathrm{ \ iterations}$' % (
        gt_segmentation.scaling_const_alpha, gt_segmentation.scaling_const_beta, gt_segmentation.sm_const,
        gt_segmentation.max_iterations), fontsize=32)

plt.setp(ax, xticks=[], yticks=[])
ax[0].imshow(gt_segmentation.image, cmap='gray')
ax[0].set_title('Image')
ax[1].imshow(contour_mask, cmap='gray')
ax[1].set_title('Output mask')

plt.show()

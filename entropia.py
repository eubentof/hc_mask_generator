import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv

import json


img_name = 'olho.jpeg'
olho = imread(img_name)
# plt.figure(num=None, figsize=(8, 6), dpi=80)
# plt.imshow(shawls)

shawl_gray = rgb2gray(img_as_ubyte(imread(img_name)))
# entropy_image = entropy(shawl_gray, disk(5))
# plt.figure(num=None, figsize=(8, 6), dpi=80)
# # plt.imshow(shawl_gray);
# plt.imshow(entropy_image, cmap='magma')


def threshold_checker(image):
    thresholds = np.arange(0.1, 1.1, 0.1)
    image_gray = rgb2gray(image)
    entropy_image = entropy(image_gray, disk(4))
    scaled_entropy = entropy_image / entropy_image.max()
    fig, ax = plt.subplots(2, 5, figsize=(17, 10))
    for n, ax in enumerate(ax.flatten()):
        ax.set_title(f'Threshold  : {round(thresholds[n],2)}',
                     fontsize=16)
        threshold = scaled_entropy > thresholds[n]
        ax.imshow(threshold, cmap='gist_stern_r')
        ax.axis('off')
    fig.tight_layout()


def apply_threshold(image, threshold=None):
    image_gray = rgb2gray(image)
    entropy_image = entropy(image_gray, disk(4))
    scaled_entropy = entropy_image / entropy_image.max()
    return scaled_entropy > threshold if threshold else scaled_entropy


def build_mask_from_threshold(image_threshold):
    size = len(image_threshold)
    curr_filter_size = len(image_threshold)/2

    quadrants_to_be_analysed = [((0, 0), image_threshold)]

    mask = {}

    def get_four_quadrants(origin, quad):
        # The origin will always be the bottom left corner of the segment
        coordX, coordY = origin
        quad = np.array(quad)
        size = len(quad)
        half = int(size/2)
        # print(f'Starting at {origin}, with size {size}, was divided in 4 quads with size {half}')
        btm_left = quad[:, :half][:half]
        top_left = quad[:, :half][half:]
        top_right = quad[:, half:][half:]
        btm_right = quad[:, half:][:half]
        return [
            ((coordX, coordY), btm_left),
            ((coordX, coordY + half), top_left),
            ((coordX + half, coordY + half), top_right),
            ((coordX + half, coordY), btm_right)
        ]

    try:

        while curr_filter_size > 1:
            # print('\n')
            # print('Current filter size:', curr_filter_size)

            sub_quads = []
            # For each quad to be analyzed, divide it into 4 quadrants
            for coords, quad in quadrants_to_be_analysed:
                sub_quads += get_four_quadrants(coords, quad)

            quadrants_to_be_analysed = []

            # print(sub_quads)
            for origin, quad in sub_quads:
                should_divide = np.isin(True, quad)
                # print(should_divide)
                if should_divide:
                    # print(f'Dividing at {origin}, with size {curr_filter_size}, into 4 quads.')
                    # print('D: ', origin, curr_filter_size)
                    # print(new_quads)
                    quadrants_to_be_analysed.append((origin, quad))
                    # print(quadrants_to_be_analysed)
                else:
                    originX, originY = origin
                    # print(f'Save mask in {origin} with size {int(curr_filter_size)}' )
                    mask[f'{originX}:{originY}'] = int(curr_filter_size)
            curr_filter_size /= 2
            # print('\n')

    except Exception as e:
        print(e)
        pass

    return mask


threshold = 0.8
image_entropy = apply_threshold(olho)
image_entropy_above_threshold = apply_threshold(olho, threshold)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5))

ax1.set_title('Entropy', fontsize=16)
ax1.imshow(image_entropy, cmap='gist_stern_r')
ax1.axis('off')
ax2.set_title(f'Threshold  : {round(threshold,2)}', fontsize=16)
ax2.imshow(image_entropy_above_threshold, cmap='gist_stern_r')
ax2.axis('off')
fig.tight_layout()
# print( x[1].__contains__(True))
mask = build_mask_from_threshold(image_entropy_above_threshold)

print(mask)

with open('masks/entropy_mask.json', 'w') as f:
    json.dump(mask, f)

# threshold_checker(olho)


plt.show()

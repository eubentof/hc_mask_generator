import numpy as np
import json

from PIL import Image, ImageOps
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2gray

from hc_mask_ploter import HCMaskPloter

def get_image_entropy(image, disk_size=4):
    image_gray = rgb2gray(image)
    entropy_image = entropy(image_gray, disk(disk_size))
    scaled_entropy = entropy_image / entropy_image.max()
    return scaled_entropy

def build_filters_from_std_threshold(entropy_image, threshold, min_filter_size=2):
    curr_filter_size = int(len(entropy_image)/2)

    quadrants_to_be_analysed = [((0, 0), entropy_image)]

    filters = {}

    def divide_quad_in_four(origin, quad):
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
        # print(btm_left.shape, top_left.shape, top_right.shape, btm_right.shape)
        # print((coordX, coordY), (coordX, coordY + half), (coordX + half, coordY + half), (coordX + half, coordY))
        subquads = [
            ((coordX, coordY), btm_left),
            ((coordX, coordY + half), top_left),
            ((coordX + half, coordY + half), top_right),
            ((coordX + half, coordY), btm_right)
        ]
        return subquads

    try:

        while curr_filter_size > 1:
            # print('=============')
            # print('Current filter size:', curr_filter_size)

            sub_quads = []
            # For each quad to be analyzed, divide it into 4 quadrants
            for coords, quad in quadrants_to_be_analysed:
                sub_quads += divide_quad_in_four(coords, quad)

            quadrants_to_be_analysed = []
            # break
            # print(sub_quads)
            for origin, quad in sub_quads:
                should_divide = np.std(quad) > threshold and curr_filter_size > min_filter_size
                # print(np.std(quad), should_divide)
                if should_divide:
                    # print(f'Dividing at {origin}, with size {curr_filter_size}, into 4 quads.')
                    # print('D: ', origin, curr_filter_size)
                    # print(new_quads)
                    quadrants_to_be_analysed.append((origin, quad))
                    # print(quadrants_to_be_analysed)
                else:
                    originX, originY = origin
                    # print(f'Save mask in {origin} with size {int(curr_filter_size)}' )
                    filters[f'{originX}:{originY}'] = int(curr_filter_size)
            curr_filter_size /= 2
            # print('\n')

    except Exception as e:
        print(e)
        pass

    return filters

file = Image.open('olho.jpeg')
data = np.asarray(file)
folder_name = 'entropy_std'

for disk_size in range(8, 17, 2):
    thresholds = np.arange(0.01, 0.1, 0.01)
    for threshold in thresholds:
        filters = []
        threshold = round(threshold,2)
        image_entropy = get_image_entropy(data, disk_size)
        filters = build_filters_from_std_threshold(image_entropy, threshold)
        print(f'Generating image for ds: {disk_size} and th: {threshold}..')

        plotter = HCMaskPloter(input_size=data.shape, filters=filters)
        flipped_image = ImageOps.flip(file)
        plotter.plot_mask(
            input_image=flipped_image, 
            mask_image=image_entropy, 
            titles={
                'plot': f'Standard Deviation Hilbert Mask - Threshhold {threshold}',
                'ax1': 'Original Image',
                'ax2': f'Entropy Disk Size: {disk_size}',
                'ax3': 'QuadTree Equivalent',
                'ax4': 'HilbertCurve',
            },
            save = True,
            filename = f'{folder_name}/ds_{disk_size}_th_{threshold}.png'
        )

        with open(f'{folder_name}/ds_{disk_size}_th_{threshold}.json', 'w') as f:
            json.dump(filters, f)
    # break

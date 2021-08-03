import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class HCMaskPloter():
    group_by_defs = {
        'mean': lambda x, y: (x + y)/2,
        'max': lambda x, y: x if x > y else y
    }

    filters_oriented = {}

    def __init__(self, input_size=(0, 0, 3), filters={}):
        self.filters = filters.copy()
        self.mask_length = int(input_size[0])
        self.input_size = input_size

        self.get_filters_oriented()

    def xy2d(self, n, x, y):
        d = 0
        s = int(n/2)
        while s > 0:
            rx = (x & s) > 0
            ry = (y & s) > 0
            d += s*s*((3*rx) ^ ry)
            s, x, y, rx, ry = self.rot(s, x, y, rx, ry)
            s = int(s / 2)
        return d

    def d2xy(self, n, d):
        t = d
        x = y = 0
        s = 1
        while s < n:
            rx = 1 & int(t/2)
            ry = 1 & (t ^ rx)
            s, x, y, rx, ry = self.rot(s, x, y, rx, ry)
            x += s*rx
            y += s*ry
            t = int(t/4)
            s *= 2
        return x, y

    def rot(self, n, x, y, rx, ry):
        if ry == 0:
            if rx == 1:
                x = n-1-x
                y = n-1-y
            x, y = y, x

        return n, x, y, rx, ry

    def get_filters_oriented(self):
        '''
            Each filter object has the { 'coordX:coordY': filter_size} format.

            The coordinates refers to the bottom left corner of the filter.

            This method calculates the other 3 corners and gets each of them is the origin of the curve in that section.

            At the end, the filters_oriented object will contain the index of the filter in the hilbert curve as key, and the coordinates, center point and length of the filter as value.

            Ex.:
            {
                256: {
                    'origin': '16:0', # the filter in the mask that starts at row 16 ans col 0
                    'center': () # that has the center point of
                    'length': 16 # and length of 16 width and 16 height
                }
            }
        '''
        self.filters_oriented = {}
        
        # Maps for each key in filters dictionary.
        for filter_key in iter(self.filters):
            # Splits the key. Ex: '0:0' -> ['0', '0']
            coords = filter_key.split(':')

            # Converts to integers and deconstruct into coordX and coordY.
            coordX, coordY = int(coords[0]), int(coords[1])

            # Gets the filter length, decreased by one, because iteration starts at 0.
            filter_length = self.filters[filter_key] - 1

            # Gets all the four corners coordinates of the filter.
            corners = [
                (coordX, coordY),  # bottom left
                (coordX + filter_length, coordY),  # top left
                (coordX + filter_length, coordY + filter_length),  # top right
                (coordX, coordY + filter_length)  # bottom right
            ]

            # List all the corresponding indexes of each corner mapped in the hilbert curve.
            corners_indexes = list(map(lambda coord: (self.xy2d(
                self.mask_length, coord[0], coord[1]), coord), corners))

            # Order the indexes, so the lowest value corresponds to the origing cordnate of the curve in that resolution.
            corners_indexes.sort()

            # Gets the origin coordinate.
            hc_origin = corners_indexes[0][0]

            # Calculate the center point of the filter.
            # Get the mid X coord
            centerX = coordX + int(filter_length/2)
            # Get the mid Y coord
            centerY = coordY + int(filter_length/2)
            center = (centerX, centerY)

            final_corners = list(map(lambda coord: coord[1], corners_indexes))

            # Saves the parameters for the filter
            self.filters_oriented[hc_origin] = {
                'coords': filter_key,
                'corners': final_corners,
                'center': center,
                'length': self.filters[filter_key]
            }
        # Sort filters by order of appearance in hilbert curve indexes.
        self.filters_oriented = dict(
            sorted(self.filters_oriented.items(), key=lambda item: item[0]))

        self.build_multi_level_hc()

    def build_multi_level_hc(self):
        self.multi_level_hc_centers = []
        self.multi_level_hc_corners = []
        for hc_origin in iter(self.filters_oriented):
            center = self.filters_oriented[hc_origin]['center']
            corners = self.filters_oriented[hc_origin]['corners']
            self.multi_level_hc_centers.append(center)
            self.multi_level_hc_corners += corners
        # print(self.multi_level_hc)

    def plot_mask(self, input_image, mask_image=None, titles={}, save=False, filename='image.png'):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))

        fig.suptitle(titles['plot'], fontsize = 16)
        
        fontsize = 12

        # Original Image
        ax1.set_title(titles['ax1'], fontsize=fontsize)
        ax1.imshow(input_image, cmap='gray')
        ax1.invert_yaxis()

        if mask_image is not None:
            ax2.set_title(titles['ax2'], fontsize=fontsize)
            ax2.imshow(mask_image, cmap='gray')
            # ax2.invert_yaxis()

        # ax1.axis('off')
        ax3.set_title(titles['ax3'], fontsize=fontsize)
        ax3.imshow(input_image, cmap='gist_stern_r')
        for filter_key in iter(self.filters):
            x, y = filter_key.split(':')
            size = self.filters[filter_key]
            ax3.add_patch(Rectangle((int(x), int(y)), size, size,
                                    linewidth=.7, edgecolor='r', facecolor="none"))

        # ax3.plot(*zip(*hc_corners), linewidth=.5)
        ax3.invert_yaxis()

        hc_centers = self.multi_level_hc_centers
        ax4.set_title(titles['ax4'], fontsize=fontsize)
        ax4.imshow(input_image, cmap='gist_stern_r')
        for filter_key in iter(self.filters):
            x, y = filter_key.split(':')
            size = self.filters[filter_key]
            ax4.add_patch(Rectangle((int(x), int(y)), size, size,
                                    linewidth=.2, edgecolor='b', facecolor="none", alpha=.5))
        ax4.plot(*zip(*hc_centers), linewidth=.7, color="r")
        ax4.invert_yaxis()
        # ax2.axis('off')
        fig.tight_layout()

        # normalizer = np.max(mask)
        # lin, col = mask.shape
        # img = np.zeros([lin, col, 1])
        # img[:, :, 0] = mask/normalizer
        # img[:, :, 1] = mask/normalizer
        # img[:, :, 2] = mask/normalizer
        # plt.imshow(img, alpha=0.5)

        if not save:
            plt.show()
        if save:
            plt.savefig(filename)
        plt.clf()
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()
        plt.close()

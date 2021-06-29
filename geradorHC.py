import numpy as np
import math
import random
# from PIL import Image as im
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
from PIL import Image


class HCMaskGenerator():
    group_by_defs = {
        'mean': lambda x, y: (x + y)/2,
        'max': lambda x, y: x if x > y else y
    }

    def __init__(self, input_size=(0, 0, 3), filters_sizes=()):
        self.filter_sizes = filters_sizes
        self.mask_size = 2 * int(input_size[0])
        self.input_size = input_size
        self.get_number_of_curves_by_level()

    def get_number_of_curves_by_level(self):
        length_of_filters = len(self.filter_sizes)
        self.amount_per_filter = []
        for i in range(length_of_filters):
            amount = self.filter_sizes[i]
            filter_size = 2**(length_of_filters - (i+1))
            self.amount_per_filter.append((filter_size, amount))
        print('Filters (filte_size, amount):', self.amount_per_filter)

    def generate_mask(self):
        mask = np.zeros(shape=(self.mask_size, self.mask_size))

        for (filter_size, amount) in self.amount_per_filter:

            if (amount == 0):  # if there is no filter with filter_size, continue to next one
                continue

            # the numbe of slices represents how many filters of filter_size can fit in the mask
            number_of_slices = int(self.mask_size/filter_size)
            mask_slices = []

            for x in range(number_of_slices):
                for y in range(number_of_slices):
                    mask_slices.append(
                        (x*filter_size, y*filter_size))

            def get_empty_spot_in_mask():
                slice = None
                while True:
                    if (len(mask_slices) == 0):
                        break

                    slice = random.choice(mask_slices)
                    mask_slices.remove(slice)

                    # check if mask in the slice coords is empty
                    if mask[slice[0]][slice[1]] == 0.:
                        break

                return slice

            for _ in range(amount):
                (lin, col) = get_empty_spot_in_mask()

                for x in range(filter_size):
                    for y in range(filter_size):
                        # filter_level = self.mask_size / filter_size
                        mask[lin + x][col + y] = filter_size
        return mask

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

    def get_hc(self, input_image,  group_by='mean'):
        mask = self.generate_mask()

        size = np.sum(self.filter_sizes)*4
        hc = np.zeros(size)
        group_def = self.group_by_defs[group_by]
        for x in range(self.mask_size):
            for y in range(self.mask_size):
                d = mask[x, y]
                p = self.xy2d(d, x, y)
                hc[p] = group_def(hc[p], input_image[x, y])
                # print(f'd:{d}, x:{x}, y:{y}, p:{xy2d(d, x, y)}')

    def plot_mask(self, save=False):
        # if (not self.mask_img):
        mask = self.generate_mask()
        print(mask)
        size = self.mask_size
        hc = np.zeros(size**4)
        for x in range(size):
            for y in range(size):

                if mask[x, y] == 0.:
                    continue

                n = int(size/mask[x, y])

                d = self.xy2d(n, x, y)
                print(d, n, x, y)
                hc[d] = n

        print(hc)
        coods_x = []
        coods_y = []

        for d in range(len(hc)):
            n = hc[d]
            px, py = self.d2xy(n, d)
            if (n == 0):
                continue
            scale = size - n + 1
            coods_x.append(px * scale)
            coods_y.append(py * scale)

        plt.plot(coods_x, coods_y)
        normalizer = np.max(mask)
        lin, col = mask.shape
        img = np.zeros([lin, col, 3])
        img[:, :, 0] = mask/normalizer
        img[:, :, 1] = mask/normalizer
        img[:, :, 2] = mask/normalizer
        plt.imshow(img)
        plt.show()


# def generateInputImage(size=256):
#     return np.random.randint(255, size=(size, size, 3), dtype=np.uint8)

image = Image.open('op.jpeg')
data = np.asarray(image)
# img = generateInputImage()
# plt.imshow(data)
# plt.show()
# print()
# input_size = data.shape
input_size = (4, 4, 3)
hc_mask = HCMaskGenerator(input_size=input_size, filters_sizes=(0, 0, 4,))
# hc_mask = HCMaskGenerator(input_size=input_size, levels=(0, 0, 2, 56))
# hc_mask = HCMaskGenerator(input_size=input_size, levels=(0, 0, 0, 64))
# 0, 0, 0, 0, 0, 0, 0, 0, 16384))
# hc_mask.getHilbertCurve(data)
hc_mask.plot_mask()

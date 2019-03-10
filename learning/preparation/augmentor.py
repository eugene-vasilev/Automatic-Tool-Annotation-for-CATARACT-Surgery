from functools import partial
import numpy as np
from PIL import Image, ImageEnhance
import cv2


class Augmentor:

    def __init__(self, angle=(-10, 10), rl_flip=True, brightness=(0.5, 2),
                 contrast=(0.5, 1.5), shift=0.1, zoom=0.1):
        self.angle = angle
        self.rl_flip = rl_flip
        self.brightness = brightness
        self.contrast = contrast
        self.shift = shift
        self.zoom = zoom

    def augment(self, img):

        if self.angle:
            rotate_angle = int(self._random(self.angle[0], self.angle[1]))
            img = self._make_probabilistic(partial(self._rotate, rotate_angle=rotate_angle), img)

        if self.tb_flip:
            img = self._make_probabilistic(self._tb_flip, img=img)

        if self.rl_flip:
            img = self._make_probabilistic(self._rl_flip, img=img)

        if self.shiift:
            y, x = img.shape[:2]
            max_dx = self.shift * x
            max_dy = self.shift * y
            shift_x, shift_y = (
                np.random.randint(-max_dx, max_dx),
                np.random.randint(-max_dy, max_dy)
            )
            img = self._make_probabilistic(partial(self._shift, shift_x=shift_x, shift_y=shift_y), img=img)

        if self.zoom:
            zoom_value = np.random.randint(-100, 100) * self.zoom / 100
            img = self._make_probabilistic(partial(self._zoom, value=zoom_value), img=img)

        if self.brightness:
            brightness_value = self._random(self.brightness[0], self.brightness[1])
            img = self._make_probabilistic(partial(self._adjust_brightness, value=brightness_value), img=img)

        if self.contrast:
            contrast_value = self._random(self.contrast[0], self.contrast[1])
            img = self._make_probabilistic(partial(self._adjust_contrast, value=contrast_value), img=img)

        return img

    def _random(self, min, max):
        return np.random.rand() * (max - min) + min

    def _make_probabilistic(self, func, img):
        if 0.5 <= np.random.random():
            return func(img)
        else:
            return img

    def _rotate(self, img, rotate_angle):
        img = Image.fromarray(img, 'RGB')
        img = img.rotate(rotate_angle, resample=Image.BILINEAR, expand=False)
        img = np.asarray(img)

        return img

    def _rl_flip(self, img):
        img = Image.fromarray(img, 'RGB')
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = np.asarray(img)

        return img

    def _shift(self, img, shift_x, shift_y):
        if shift_x > 0:
            img[:, shift_x:] = img[:, :-shift_x]
            img[:, :shift_x] = 0
        elif shift_x < 0:
            img[:, :shift_x] = img[:, -shift_x:]
            img[:, shift_x:] = 0

        if shift_y > 0:
            img[shift_y:, :] = img[:-shift_y, :]
            img[:shift_y, :] = 0
        elif shift_y < 0:
            img[:shift_y, :] = img[-shift_y:, :]
            img[shift_y:, :] = 0

        return img

    def _zoom(self, img, value):
        img_height, img_width, = img.shape[:2]
        height_zoom = img_height * value // 2
        width_zoom = img_width * value // 2
        if value > 0:
            img = img[height_zoom: -height_zoom, width_zoom: -width_zoom]
        elif value < 0:
            img = np.pad(img, ((height_zoom, height_zoom), (width_zoom, width_zoom)),
                         'constant', constant_values=((0, 0), (0, 0)))
            img = cv2.resize(img, (img_width, img_height))

        return img

    def _adjust_brightness(self, img, value):
        prepared_img = Image.fromarray(img, 'RGB')

        enhancer = ImageEnhance.Brightness(prepared_img)
        prepared_img = enhancer.enhance(value)

        prepared_img = np.asarray(prepared_img)
        img = np.dstack([prepared_img, img])

        return img

    def _adjust_contrast(self, img, value):
        prepared_img = Image.fromarray(img, 'RGB')

        enhancer = ImageEnhance.Contrast(prepared_img)
        prepared_img = enhancer.enhance(value)

        prepared_img = np.asarray(prepared_img)
        img = np.dstack([prepared_img, img])

        return img

from functools import partial
import numpy as np
from PIL import Image, ImageEnhance
from datetime import datetime


class Augmentor:

    def __init__(self, angle=(-10, 10), rl_flip=True, brightness=(0.5, 2),
                 contrast=(0.5, 1.5), shift=0.1, zoom=0.1):
        self.angle = angle
        self.rl_flip = rl_flip
        self.brightness = brightness
        self.contrast = contrast
        self.shift = shift

    def augment(self, img):
        np.random.seed(seed=datetime.now().microsecond)
        if self.shift:
            y, x = img.shape[:2]
            max_dx = self.shift * x
            max_dy = self.shift * y
            shift_x, shift_y = (
                np.random.randint(-max_dx, max_dx),
                np.random.randint(-max_dy, max_dy)
            )
            img = self._make_probabilistic(partial(self._shift, shift_x=shift_x, shift_y=shift_y), img=img)
        if self.angle:
            rotate_angle = int(self._random(self.angle[0], self.angle[1]))
            img = self._make_probabilistic(partial(self._rotate, rotate_angle=rotate_angle), img)

        if self.rl_flip:
            img = self._make_probabilistic(self._rl_flip, img=img)

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
        new_img = img = Image.fromarray(img, 'RGB')
        new_img = new_img.rotate(rotate_angle, resample=Image.BILINEAR, expand=False)
        new_img = np.asarray(new_img)

        return new_img

    def _rl_flip(self, img):
        new_img = Image.fromarray(img, 'RGB')
        new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
        new_img = np.asarray(new_img)

        return new_img

    def _shift(self, img, shift_x, shift_y):
        new_img = img.copy()
        if shift_x > 0:
            new_img[:, shift_x:] = img[:, :-shift_x]
            new_img[:, :shift_x] = 0
        elif shift_x < 0:
            new_img[:, :shift_x] = img[:, -shift_x:]
            new_img[:, shift_x:] = 0

        if shift_y > 0:
            new_img[shift_y:, :] = img[:-shift_y, :]
            new_img[:shift_y, :] = 0
        elif shift_y < 0:
            new_img[:shift_y, :] = img[-shift_y:, :]
            new_img[shift_y:, :] = 0

        return new_img

    def _adjust_brightness(self, img, value):
        prepared_img = Image.fromarray(img, 'RGB')

        enhancer = ImageEnhance.Brightness(prepared_img)
        prepared_img = enhancer.enhance(value)

        new_img = np.asarray(prepared_img)

        return new_img

    def _adjust_contrast(self, img, value):
        prepared_img = Image.fromarray(img, 'RGB')

        enhancer = ImageEnhance.Contrast(prepared_img)
        prepared_img = enhancer.enhance(value)

        new_img = np.asarray(prepared_img)

        return new_img

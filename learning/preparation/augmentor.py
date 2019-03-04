from functools import partial
import numpy as np
from PIL import Image, ImageEnhance


class Augmentor:

    def __init__(self, angle=(-180, 180), rl_flip=True, tb_flip=True, gamma=(0.5, 1.5), brightness=(0.5, 2),
                 contrast=(0.5, 1.5), hue=(-0.03, 0.03), saturation=(0.5, 1.5), color_augment_count=2):
        self.angle = angle
        self.rl_flip = rl_flip
        self.tb_flip = tb_flip
        self.gamma = gamma
        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue
        self.saturation = saturation
        self.color_augment_count = color_augment_count

    def augment(self, img, mask):
        mask = np.squeeze(mask, axis=(2,))

        if self.angle:
            rotate_angle = int(self._random(self.angle[0], self.angle[1]))
            img, mask = self._make_probabilistic(partial(self._rotate, rotate_angle=rotate_angle), img, mask)

        if self.tb_flip:
            img, mask = self._make_probabilistic(self._tb_flip, img=img, mask=mask)

        if self.rl_flip:
            img, mask = self._make_probabilistic(self._rl_flip, img=img, mask=mask)

        color_augmentations = np.array([[self.gamma, self._adjust_gamma],
                                        [self.brightness, self._adjust_brightness],
                                        [self.contrast, self._adjust_contrast],
                                        [self.hue, self._adjust_hue],
                                        [self.saturation, self._adjust_saturation]
                                        ])

        color_augment_logs = []
        for _ in range(self.color_augment_count):
            augment_index = np.random.randint(5)
            if augment_index not in color_augment_logs:
                color_augment_logs.append(augment_index)
                augment_range, augment_func = color_augmentations[augment_index]
                value = self._random(augment_range[0], augment_range[1])
                img, mask = self._make_probabilistic(partial(augment_func, value=value), img=img, mask=mask)

        return img, mask

    def _random(self, min, max):
        return np.random.rand() * (max - min) + min

    def _make_probabilistic(self, func, img, mask):
        if 0.5 <= np.random.random():
            return func(img, mask)
        else:
            return img, mask

    def _rotate(self, img, mask, rotate_angle):
        img = Image.fromarray(img, 'RGBA')
        mask = Image.fromarray(mask)

        img = img.rotate(rotate_angle, resample=Image.BILINEAR, expand=False)
        mask = mask.rotate(rotate_angle, resample=Image.BILINEAR, expand=False)

        img = np.asarray(img)
        mask = np.asarray(mask)

        return img, mask

    def _rl_flip(self, img, mask):
        img = Image.fromarray(img, 'RGBA')
        mask = Image.fromarray(mask)

        img, mask = img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)

        img = np.asarray(img)
        mask = np.asarray(mask)

        return img, mask

    def _tb_flip(self, img, mask):
        img = Image.fromarray(img, 'RGBA')
        mask = Image.fromarray(mask)

        img, mask = img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)

        img = np.asarray(img)
        mask = np.asarray(mask)

        return img, mask

    def _adjust_gamma(self, img, mask, value):
        prepared_img = img[:, :, :3]
        prepared_img = Image.fromarray(prepared_img, 'RGB')

        gain = 1
        input_mode = prepared_img.mode
        gamma_map = [255 * gain * pow(ele / 255., value) for ele in range(256)] * 3
        prepared_img = prepared_img.point(gamma_map)

        prepared_img = prepared_img.convert(input_mode)

        prepared_img = np.asarray(prepared_img)
        img = np.dstack([prepared_img, img[..., 3]])

        return img, mask

    def _adjust_brightness(self, img, mask, value):
        prepared_img = img[:, :, :3]
        prepared_img = Image.fromarray(prepared_img, 'RGB')

        enhancer = ImageEnhance.Brightness(prepared_img)
        prepared_img = enhancer.enhance(value)

        prepared_img = np.asarray(prepared_img)
        img = np.dstack([prepared_img, img[..., 3]])

        return img, mask

    def _adjust_contrast(self, img, mask, value):
        prepared_img = img[:, :, :3]
        prepared_img = Image.fromarray(prepared_img, 'RGB')

        enhancer = ImageEnhance.Contrast(prepared_img)
        prepared_img = enhancer.enhance(value)

        prepared_img = np.asarray(prepared_img)
        img = np.dstack([prepared_img, img[..., 3]])

        return img, mask

    def _adjust_hue(self, img, mask, value):
        prepared_img = img[:, :, :3]
        prepared_img = Image.fromarray(prepared_img, 'RGB')

        if not(-0.5 <= value <= 0.5):
            raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(value))
        input_mode = prepared_img.mode
        h, s, v = prepared_img.convert('HSV').split()
        np_h = np.array(h, dtype=np.uint8)
        np_h += np.uint8(value * 255)
        h = Image.fromarray(np_h, 'L')
        prepared_img = Image.merge('HSV', (h, s, v)).convert(input_mode)

        prepared_img = np.asarray(prepared_img)
        img = np.dstack([prepared_img, img[..., 3]])

        return img, mask

    def _adjust_saturation(self, img, mask, value):
        prepared_img = img[:, :, :3]
        prepared_img = Image.fromarray(prepared_img, 'RGB')

        enhancer = ImageEnhance.Color(prepared_img)
        prepared_img = enhancer.enhance(value)

        prepared_img = np.asarray(prepared_img)
        img = np.dstack([prepared_img, img[..., 3]])

        return img, mask

import cv2
import numpy as np
from preparation.augmentor import Augmentor
from preparation.utils import get_snake_case, get_class_from_path
import pandas as pd
import os
from multiprocessing.pool import ThreadPool as Pool
from glob import glob


class Processor:

    def __init__(self, batch_size, width, height):
        self.batch_size = batch_size
        self.width = width
        self.height = height
        columns = pd.read_csv('data/train_labels/train01.csv').columns[1:]
        columns = list(map(lambda x: get_snake_case(x), columns))
        columns_to_index = {column_name: index for (index, column_name) in enumerate(columns)}
        columns_to_index.update({'no_tools': 21})
        self.columns = columns_to_index

    def process(self, imgs_paths, augment=True):
        new_imgs = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        new_labels = np.zeros((self.batch_size, 22), dtype=np.float32)
        if not len(imgs_paths):
            return new_imgs, new_labels

        for i in range(0, len(imgs_paths)):
            img = cv2.imread(imgs_paths[i], 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Augmentor().augment(img) if augment else img
            new_imgs[i] = img
            current_class = get_class_from_path(imgs_paths[i])
            new_labels[i][self.columns[current_class]] = 1.

        new_imgs /= 255

        return new_imgs, new_labels

    def delete_empty_files(self, imgs_paths, folder_path):

        def delete_empty_file(path):
            img = cv2.imread(path)
            shape = img.shape[:2]
            if not shape[0] or not shape[1]:
                os.remove(path)

        with Pool(processes=12) as pool:
            pool.map(delete_empty_file, imgs_paths)

        return np.array(glob(folder_path))

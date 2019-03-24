# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
import cv2
import argparse
from glob import glob
from multiprocessing.pool import ThreadPool as Pool
import numpy as np
from preparation.utils import create_dir, remove_dir


class HogExtractor:

    def __init__(self, width=480, height=270):
        self.hog = cv2.HOGDescriptor()
        self.width = width
        self.height = height

    def get_hog_descriptor(self, path):
        img = cv2.imread(path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.width, self.height))

        new_file_name = path.replace('extracted_frames', 'hog_descs')
        new_file_name = new_file_name.replace('jpg', 'npy')
        folder = new_file_name[:new_file_name.rfind('/')]
        create_dir(folder)

        np.save(new_file_name, self.hog.compute(img, (60, 60), (15, 15)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--workers', type=int, default=12,
                        help='Number of processes')

    args = parser.parse_args()

    train_imgs_folder = 'learning/data/extracted_frames/*/*/*.jpg'
    train_imgs_paths = glob(train_imgs_folder)

    hog_extractor = HogExtractor()

    remove_dir('learning/data/hog_descs')

    with Pool(processes=args.workers) as pool:
        pool.map(hog_extractor.get_hog_descriptor, train_imgs_paths)

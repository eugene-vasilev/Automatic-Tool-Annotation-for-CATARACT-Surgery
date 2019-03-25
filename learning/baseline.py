from model.svm import SVM
# from sklearn.model_selection import GridSearchCV
import cv2
import argparse
from glob import glob
from multiprocessing.pool import ThreadPool as Pool
import numpy as np
import cupy as cp
import pandas as pd
import gc
from preparation.utils import create_dir, remove_dir, get_snake_case

columns = pd.read_csv('./learning/data/train_labels/train01.csv').columns[1:]
columns = list(map(lambda x: get_snake_case(x), columns))
columns_to_index = {column_name: index for (index, column_name) in enumerate(columns)}
columns_to_index.update({'no_tools': 21})


def read_npy(path):
    img = np.load(path)
    end_index = path.rfind('/')
    start_index = path[:end_index].rfind('/') + 1
    label = columns_to_index[path[start_index: end_index]]
    return img, label


def get_cupy_split(train_paths, test_paths, workers):
    with Pool(processes=workers) as pool:
        train_set = pool.map(read_npy, train_paths)
        test_set = pool.map(read_npy, test_paths)

    return train_set
    '''
    print(len(train_set), len(train_set[0]), len(train_set[1]))

    train_descs, train_labels = train_set[:, 0], train_set[:, 1]
    test_descs, test_labels = test_set[:, 0], test_set[:, 1]

    train_descs, train_labels = cp.array(train_descs), cp.array(train_labels)
    test_descs, test_labels = cp.array(test_descs), cp.array(test_labels)

    gc.collect()

    return train_descs, train_labels, test_descs, test_labels
    '''

def train_and_compute_misclassification(kernel, kernel_params, classification_strategy, x_train, y_train,
                                        x_test, y_test, lambduh=1, use_optimal_lambda=False,
                                        n_folds=3, max_iter=200):
    print('svm-gpu, {} kernel, parameters {}'.format(kernel, kernel_params))
    svm = SVM(kernel, kernel_params, lambduh, max_iter, classification_strategy, x=x_train, y=y_train,
              n_folds=n_folds, display_plots=True)

    svm.fit(x_train, y_train, use_optimal_lambda=use_optimal_lambda)
    if svm._classification_strategy == 'binary':
        svm.plot_misclassification_error()

    misclassification_error = svm.compute_misclassification_error(x_test, y_test)
    print('Misclassification error (test), {}, {}lambda = {} : {}\n'.format(svm._classification_strategy,
          ('optimal ' if use_optimal_lambda else ''), svm._lambduh, misclassification_error))


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

        np.save(new_file_name, self.hog.compute(img, (230, 230), (15, 15)))


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

    del train_imgs_paths

    train_descs_folder = 'learning/data/hog_descs/train/*/*.npy'
    validation_descs_folder = 'learning/data/hog_descs/validation/*/*.npy'
    train_descs_paths = glob(train_descs_folder)
    validation_descs_paths = glob(validation_descs_folder)

    x_set = get_cupy_split(train_descs_paths, validation_descs_paths, args.workers)

    #train_and_compute_misclassification('rbf', {'sigma': 7}, 'ovr', x_train, y_train, x_test, y_test,
                                        #use_optimal_lambda=True)

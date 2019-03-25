from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
import cv2
import argparse
from glob import glob
from multiprocessing.pool import ThreadPool as Pool
import numpy as np
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


def get_descs_split(train_paths, test_paths, workers):
    with Pool(processes=workers) as pool:
        train_set = pool.map(read_npy, train_paths)
        test_set = pool.map(read_npy, test_paths)

    train_descs = [np.reshape(x[0], x[0].shape[:-1]) for x in train_set]
    test_descs = [np.reshape(x[0], x[0].shape[:-1]) for x in test_set]
    train_labels = [x[1] for x in train_set]
    test_labels = [x[1] for x in test_set]

    train_descs, train_labels = np.array(train_descs), np.array(train_labels)
    test_descs, test_labels = np.array(test_descs), np.array(test_labels)
    gc.collect()

    return train_descs, train_labels, test_descs, test_labels


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

        np.save(new_file_name, self.hog.compute(img, (450, 450), (15, 15)))


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

    x_train, y_train, x_test, y_test = get_descs_split(train_descs_paths, validation_descs_paths, args.workers)

    clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=15))

    y_score = clf.fit(x_train, y_train).decision_function(x_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(22):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

#    Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

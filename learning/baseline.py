from sklearn.ensemble import ExtraTreesClassifier
import cv2
import argparse
from glob import glob
from multiprocessing.pool import ThreadPool as Pool
import numpy as np
import pandas as pd
import gc
from preparation.utils import create_dir, remove_dir, get_snake_case
from functools import partial

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


def get_hog_descs(train_paths, workers):
    with Pool(processes=workers) as pool:
        train_set = pool.map(read_npy, train_paths)

    train_descs = [np.reshape(x[0], x[0].shape[:-1]) for x in train_set]
    train_labels = [x[1] for x in train_set]

    train_descs, train_labels = np.array(train_descs), np.array(train_labels)
    gc.collect()

    return train_descs, train_labels


def save_test_predictions(model):
    test_video_paths = sorted(glob('./learning/data/test/*.mp4'))
    hog_extractor = HogExtractor()

    for video_path in test_video_paths:
        df_columns = list(pd.read_csv('./learning/data/train_labels/train01.csv').columns)
        result_df = pd.DataFrame(columns=df_columns)

        video = cv2.VideoCapture(video_path)
        success, image = video.read()
        num = 1
        success = True
        while success:
            success, img = video.read()

            if hasattr(img, 'shape'):
                img = cv2.resize(img, (hog_extractor.width, hog_extractor.height))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                hog_descriptor = hog_extractor.get_hog_descriptor(img, save=False)
                hog_descriptor = np.reshape(hog_descriptor, (1, hog_descriptor.shape[0]))
                predictions = model.predict_proba(hog_descriptor)
                predictions = np.delete(predictions, len(predictions[0]) - 1, None)
            else:
                predictions = np.array([0. for _ in range(21)])

            predictions = np.insert(predictions, 0, num)
            result_df.loc[num - 1] = predictions
            num += 1
        folder = './learning/predictions/{}/{}/'.format('baseline', 'test')
        create_dir(folder)
        save_path = '{}{}'.format(folder, video_path[video_path.rfind('/') + 1:])
        save_path = save_path.replace('mp4', 'csv')
        result_df.to_csv(save_path, header=False, index=False)


class HogExtractor:

    def __init__(self, width=480, height=270):
        self.hog = cv2.HOGDescriptor()
        self.width = width
        self.height = height

    def get_hog_descriptor(self, img_source, save=True):
        if isinstance(img_source, str):
            img = cv2.imread(img_source, 1)
            img = cv2.resize(img, (self.width, self.height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif hasattr(img_source, 'shape'):
            assert img_source.shape == (self.height, self.width, 3,)
            img = img_source
        hog_descriptor = self.hog.compute(img, (450, 450), (15, 15))
        if not save:
            return hog_descriptor

        new_file_name = img_source.replace('extracted_frames', 'hog_descs')
        new_file_name = new_file_name.replace('jpg', 'npy')
        folder = new_file_name[:new_file_name.rfind('/')]
        create_dir(folder)

        np.save(new_file_name, hog_descriptor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--workers', type=int, default=12,
                        help='Number of processes')
    parser.add_argument('-s', '--step', type=int, default=2,
                        choices=[0, 1, 2],
                        help='Step of main to run: 0=all, 1=Extract&Save hog descs, 2=Train baseline/test predict')

    args = parser.parse_args()

    if not args.step or args.step % 2:
        train_imgs_folder = 'learning/data/extracted_frames/*/*/*.jpg'
        train_imgs_paths = glob(train_imgs_folder)

        hog_extractor = HogExtractor()

        remove_dir('learning/data/hog_descs')

        with Pool(processes=args.workers) as pool:
            pool.map(partial(hog_extractor.get_hog_descriptor, save=True), train_imgs_paths)

        del train_imgs_paths

    if not args.step or not args.step % 2:
        train_descs_folder = 'learning/data/hog_descs/*/*/*.npy'
        train_descs_paths = glob(train_descs_folder)

        x_train, y_train = get_hog_descs(train_descs_paths, args.workers)

        et_clf = ExtraTreesClassifier(n_estimators=100, n_jobs=-1)

        et_clf = et_clf.fit(x_train, y_train)

        save_test_predictions(et_clf)

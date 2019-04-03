import os
import shutil
from glob import glob


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception:
            pass
        print('Folder is created: {}'.format(path))


def get_folder_data(glob_path):
    data_list = glob(glob_path)
    data_list = list(map(lambda x: x[x.rfind('/') + 1:x.rfind('.')], data_list))
    return data_list


def get_snake_case(string):
    string = string.lower()
    string = string.replace(' ', '_')
    string = string.replace('/', '_')
    return string


def get_class_from_path(path):
    end = path.rfind('/')
    start = path[:end].rfind('/') + 1
    return path[start:end]


def remove_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)

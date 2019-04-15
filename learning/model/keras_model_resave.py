from keras.models import load_model
from glob import glob
from metrics import auc, precision, recall, f1


def save_json(model, path):
    model_json = model.to_json()
    with open(path, "w") as json_file:
        json_file.write(model_json)


def save_weights(model, path):
    model.save_weights(path)


def resave_model(model_path, save_path):
    model = load_model(model_path, custom_objects={"auc": auc,
                                                   "precision": precision,
                                                   "recall": recall,
                                                   "f1": f1})
    save_json(model, save_path + '/model.json')
    save_weights(model, save_path + '/model.h5')

if __name__ == '__main__':
    model_folders = glob('./model/saved_models/*')
    for model_folder in model_folders:
        models = sorted(glob(model_folder + '/*.hdf5'))
        last_model = models[-1]
        resave_model(last_model, model_folder)
        model_name = model_folder[model_folder.rfind('/') + 1:]
        print('Model {} resaved!'.format(model_name))

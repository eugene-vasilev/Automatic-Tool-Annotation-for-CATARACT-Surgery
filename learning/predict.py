import pandas as pd
import cv2
import numpy as np
import argparse
from glob import glob
from keras.models import model_from_json
from pandas import read_csv
from sklearn.metrics import roc_curve, auc
from preparation.utils import create_dir


def load_model(model_folder):
    model_path = model_folder + '/model.json'
    weights_path = model_folder + '/model.h5'
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)

    return model


def calculate_auc(original_csv, predicted_csv):
    try:
        for tool in range(1, 22):

            truth = []
            predictions = []

            # parsing the right column for the current tool
            truth_data = read_csv(original_csv, header=0, skipinitialspace=True,
                                  usecols=[tool], squeeze=True, dtype='float32').tolist()
            prediction_data = read_csv(predicted_csv, header=None, skipinitialspace=True,
                                       usecols=[tool], squeeze=True, dtype='float32').tolist()
            if len(truth_data) != len(prediction_data):
                raise ValueError('Files {} and {} have different row counts'.format(original_csv, predicted_csv))

            # appending rows with consensual ground truth
            indices = [index for index, value in enumerate(truth_data) if value != 0.5]
            truth += [truth_data[index] for index in indices]
            predictions += [prediction_data[index] for index in indices]

            # computing the area under the ROC curve
            fpr, tpr, _ = roc_curve(truth, predictions)
            score = auc(fpr, tpr)
            print('Tool {} : AUC = {}'.format(tool, score))
    except Exception as e:
        print('Error: missing column in {} for tool number {}!'.format('this_file', tool)
              if 'Usecols' in str(e) else 'Error: {}!'.format(e))


def construct_predictions_dataframe(video_path, model, delimiter, model_name):
    video = cv2.VideoCapture(video_path)
    success, image = video.read()
    num = 1
    success = True

    df_columns = list(pd.read_csv('./data/train_labels/train01.csv').columns)
    result_df = pd.DataFrame(columns=df_columns)

    while success:
        success, img = video.read()

        if hasattr(img, 'shape'):
            img = cv2.resize(img, (img.shape[1] // delimiter, img.shape[0] // delimiter))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)
            img /= 255
            img = np.expand_dims(img, axis=0)
            predictions = list(model.predict(img)[0][:-1])
        else:
            predictions = [0. for _ in range(21)]

        predictions.insert(0, num)
        result_df.loc[num - 1] = predictions
        num += 1
    folder = './predictions/{}/'.format(model_name)
    create_dir(folder)
    save_path = '{}{}'.format(folder, video_path[video_path.rfind('/') + 1:])
    save_path = save_path.replace('mp4', 'csv')
    result_df.to_csv(save_path, header=False, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-mf', '--model_folder', type=str,
                        required=True, help='Path to folder that contains model.h5 and model.json files')
    parser.add_argument('-d', '--delimiter',
                        type=int, default=4,
                        help='Delimiter for image height and width')

    args = parser.parse_args()

    args.model_folder = args.model_folder[:-1] if args.model_folder[-1] == '/' else args.model_folder

    test_video_paths = sorted(glob('./data/test/*.mp4'))
    train_video_paths = sorted(glob('./data/train/*.mp4'))

    model = load_model(args.model_folder)

    model_name = args.model_folder[args.model_folder.rfind('/') + 1:]

    for i in train_video_paths:
        construct_predictions_dataframe(i, model, args.delimiter, model_name)

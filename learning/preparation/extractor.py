import pandas as pd
from preparation.utils import create_dir, get_snake_case, remove_dir
import cv2
from glob import glob
import argparse
from multiprocessing.pool import ThreadPool as Pool
from functools import partial


def get_tools_frequency_split(frequency_threshold):
    csv_paths = glob('data/train_labels/*.csv')
    tools_frequency = {}
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df.columns = list(map(lambda x: get_snake_case(x), df.columns))
        if not len(tools_frequency):
            for column_name in df.columns[1:]:
                tools_frequency.update({column_name: 0})

        for column_name in df.columns[1:]:
            tools_frequency.update({column_name: tools_frequency[column_name] + df[df[column_name] == 1].shape[0]})

    rare_tools = [x for x in tools_frequency.keys() if tools_frequency[x] < frequency_threshold]
    common_tools = [x for x in tools_frequency.keys() if tools_frequency[x] >= frequency_threshold]

    return rare_tools, common_tools, tools_frequency


def extract_frames_from_video(files, rare_tools_threshold, common_tools_step,
                              no_tools_frames_count, column_names=None, delimiter=4):
    csv_path, video_path = files
    video_name = video_path[video_path.rfind('/') + 1:]
    print('{} in processing'.format(video_name))
    validation_nums = ['04', '12', '21']

    df = pd.read_csv(csv_path)
    df.columns = list(map(lambda x: get_snake_case(x), df.columns))

    num_end = video_path.rfind('.')
    video_num = video_path[num_end - 2: num_end]
    split_name = 'validation' if video_num in validation_nums else 'train'

    rare_tools, common_tools, _ = get_tools_frequency_split(frequency_threshold=rare_tools_threshold)

    df_tools = df[df[df.columns[1:]].sum(axis=1) >= 1]
    df_no_tools = df[df[df.columns[1:]].sum(axis=1) == 0]
    no_tools_step = int(df_no_tools.shape[0] / no_tools_frames_count)

    frames_labels = {}

    if column_names:
        for column_name in column_names:
            assert column_name in df_tools.columns
        target_columns = column_names
    else:
        target_columns = df_tools.columns[1:]

    for column_name in target_columns:
        folder_path = 'data/extracted_frames/{}/{}/'.format(split_name, column_name)
        df_this_tool = df_tools[df_tools[column_name] == 1]
        create_dir(folder_path)
        frames = df_this_tool['frame']
        if column_name in rare_tools:
            for frame in frames:
                frames_labels.update({int(frame): column_name})
        elif column_name in common_tools:
            for frame in frames[::common_tools_step]:
                frames_labels.update({int(frame): column_name})

    create_dir('data/extracted_frames/{}/no_tools/'.format(split_name))
    for frame in df_no_tools['frame'][::no_tools_step]:
        frames_labels.update({int(frame): 'no_tools'})

    video = cv2.VideoCapture(video_path)
    success, image = video.read()
    num = 1
    success = True

    frame_nums = list(frames_labels.keys())

    while success:
        success, image = video.read()
        if num in frame_nums:
            if hasattr(image, 'shape'):
                frame_nums.remove(num)
                folder_path = 'data/extracted_frames/{}/{}/'.format(split_name, frames_labels[num])
                write_path = folder_path + '{}_{}.jpg'.format(video_num, num)
                image = cv2.resize(image, (image.shape[1] // delimiter, image.shape[0] // delimiter))
                cv2.imwrite(write_path, image)
        num += 1


def extract_frames(args):
    remove_dir('data/extracted_frames/')
    videos = glob('data/train/*.mp4')
    csvs = glob('data/train_labels/*.csv')
    videos.sort()
    csvs.sort()
    videos_length = len(videos)
    process_function = partial(extract_frames_from_video,
                               rare_tools_threshold=args.rare_tools_threshold,
                               common_tools_step=args.common_tools_step,
                               no_tools_frames_count=args.no_tools_frames_count // videos_length,
                               column_names=args.column_names,
                               delimiter=args.delimiter)

    with Pool(processes=args.workers) as pool:
        pool.map(process_function, zip(csvs, videos))

    print('Process finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--rare_tools_threshold',
                        type=int, default=2000,
                        help='Threshold in tools total count for splitting common/rare tools')
    parser.add_argument('-s', '--common_tools_step',
                        type=int, default=6,
                        help='Frequency step for common tools')
    parser.add_argument('-fc', '--no_tools_frames_count',
                        type=int, default=40000,
                        help='Total count of non-tool frames in result dataset')
    parser.add_argument('-cn', '--column_names',
                        type=str, nargs='*',
                        help='Column names for only this columns processing')
    parser.add_argument('-w', '--workers',
                        type=int, default=12,
                        help='Number of processes')
    parser.add_argument('-d', '--delimiter',
                        type=int, default=4,
                        help='Delimiter for image height and width')

    args = parser.parse_args()

    extract_frames(args)

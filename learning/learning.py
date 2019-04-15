from model.network_base import darknet19, darknet53, resnet50, mobilenetv2, inceptionv3
from model.metrics import auc, precision, recall, f1
from model.callbacks import make_callbacks
import argparse
from glob import glob
from preparation.multiprocessing_generator import Seq
from keras.utils import multi_gpu_model
from keras.optimizers import Adam

from keras.losses import categorical_crossentropy
from datetime import datetime

from preparation.processing import Processor
from functools import partial


def validation_evaluate(model, width=480, height=270, batch=32, min_lr=0.000001, workers=12):
    _, validation_seq = get_generators(width, height, batch)

    assert model.input_shape[1:] == (height, width, 3)
    assert model.output_shape[1:] == (22,)

    optimizer = Adam(lr=min_lr)
    model.compile(optimizer=optimizer,
                  loss=categorical_crossentropy,
                  metrics=[auc, precision, recall, f1, 'acc']
                  )

    return model.evaluate_generator(validation_seq, len(validation_seq), workers=workers,
                                    use_multiprocessing=False, verbose=1), model.metrics_names


def get_generators(width, height, batch):
    train_imgs_folder = 'data/extracted_frames/train/*/*.jpg'
    train_imgs_paths = glob(train_imgs_folder)
    validation_imgs_folder = 'data/extracted_frames/validation/*/*.jpg'
    validation_imgs_paths = glob(validation_imgs_folder)

    processor = Processor(batch, width, height)
    train_imgs_paths = processor.delete_empty_files(train_imgs_paths, train_imgs_folder)
    validation_imgs_paths = processor.delete_empty_files(validation_imgs_paths, validation_imgs_folder)

    print('Make generators')

    train_seq = Seq(train_imgs_paths, batch, processor.process)
    validation_seq = Seq(validation_imgs_paths, batch, partial(processor.process, augment=False))

    return train_seq, validation_seq


def train(model='darknet19', width=480, height=270, batch=32, tensorboard=False, max_lr=0.001, min_lr=0.000001,
          workers=12, multi_gpu=1, epochs=500, steps_multiplier=4, distributed=False, loaded_model=None):
    if distributed:
        import tensorflow as tf
        from keras import backend as K
        import horovod.keras as hvd
        hvd.init()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        K.set_session(tf.Session(config=config))

        min_lr *= hvd.size()
        max_lr *= hvd.size()

    train_seq, validation_seq = get_generators(width, height, batch)
    train_steps = len(train_seq)
    validation_steps = len(validation_seq)

    print('Build and compile model')
    if loaded_model:
        assert loaded_model.input_shape[1:] == (height, width, 3)
        assert loaded_model.output_shape[1:] == (22,)
        model = loaded_model
    else:
        if model == 'darknet19':
            model = darknet19((height, width, 3), 22)
        elif model == 'darknet53':
            model = darknet53((height, width, 3), 22)
        elif model == 'resnet50':
            model = resnet50((height, width, 3), 22)
        elif model == 'mobilenetv2':
            model = mobilenetv2((height, width, 3), 22)
        elif model == 'inceptionv3':
            model = inceptionv3((height, width, 3), 22)

    if multi_gpu > 1:
        model = multi_gpu_model(model, gpus=multi_gpu)

    model.summary()

    optimizer = Adam(lr=min_lr)

    if distributed:
        optimizer = hvd.DistributedOptimizer(optimizer)
        train_steps //= hvd.size()
        validation_steps //= hvd.size()

    model.compile(optimizer=optimizer,
                  loss=categorical_crossentropy,
                  metrics=[auc, precision, recall, f1, 'acc']
                  )

    print('Make callbacks')

    current_datetime = datetime.now()
    model_name = '{}_{}-{}-{}_{}:{}:{}_{}x{}'.format(model, current_datetime.day, current_datetime.month,
                                                     current_datetime.year, current_datetime.hour,
                                                     current_datetime.minute, current_datetime.second,
                                                     height, width)

    callbacks = make_callbacks(model_name, min_lr, max_lr, train_steps * steps_multiplier,
                               tensorboard, save=not bool(loaded_model))

    if distributed:
        callbacks.insert(0, hvd.callbacks.MetricAverageCallback())
        callbacks.insert(0, hvd.callbacks.BroadcastGlobalVariablesCallback(0))

    model.fit_generator(
        generator=train_seq,
        steps_per_epoch=train_steps,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_seq,
        validation_steps=validation_steps,
        use_multiprocessing=True,
        workers=workers
    )

    if loaded_model:
        return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, default='darknet53',
                        choices=['darknet19', 'darknet53', 'resnet50', 'mobilenetv2', 'inceptionv3'],
                        help='CNN architecture')
    parser.add_argument('--width', type=int, default=480,
                        help='Image width')
    parser.add_argument('--height', type=int, default=270,
                        help='Image height')
    parser.add_argument('-b', '--batch', type=int, default=32,
                        help='Batch size')
    parser.add_argument('-t', '--tensorboard', action='store_true', default=False,
                        help='Use tensorboard callback')
    parser.add_argument('--max_lr', type=float, default=0.001,
                        help='Max learning rate for Cyclic LR')
    parser.add_argument('--min_lr', type=float, default=0.000001,
                        help='Min learning rate for Cyclic LR')
    parser.add_argument('--workers', type=int, default=12,
                        help='Number of processes')
    parser.add_argument('--multi_gpu', type=int, default=1,
                        help='Choose the gpu count')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs')
    parser.add_argument('--steps_multiplier', type=int, default=4,
                        help='Multiplier value for Cyclic LR')
    parser.add_argument('--distributed', action='store_true', default=False,
                        help='Needed if run in distributed mode (horovod using)')

    args = parser.parse_args()

    print("")
    print('Arguments is valid')

    train(model=args.model, width=args.width, height=args.height, batch=args.batch, tensorboard=args.tensorboard,
          max_lr=args.max_lr, min_lr=args.min_lr, workers=args.workers, multi_gpu=args.multi_gpu,
          epochs=args.epochs, steps_multiplier=args.steps_multiplier, distributed=args.distributed)

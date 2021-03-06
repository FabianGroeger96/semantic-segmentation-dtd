import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomContrast, RandomTranslation
from pathlib import Path

from src.settings.settings import Settings


class DTDDataset:
    """
    Dataset representation for the Describable Textures Dataset (DTD).
    Link to dataset: https://www.robots.ox.ac.uk/~vgg/data/dtd/
    Implemented as Singleton.
    """

    # instance of the class
    __instance = None

    @staticmethod
    def get_instance(settings: Settings, log: bool = False):
        """ Static access method. """
        if DTDDataset.__instance is None:
            DTDDataset(settings=settings, log=log)

        return DTDDataset.__instance

    def __init__(self, settings: Settings,
                 log: bool = False, name: str = 'DTD'):
        """ Virtually private constructor. """
        # throw exception if at initialization an instance already exists
        if DTDDataset.__instance is not None:
            raise Exception('Dataset should be a singleton \
                            and instance is not None at initialization.')
        else:
            DTDDataset.__instance = self

        # parameters
        self.log = log
        self.name = name
        self.settings = settings
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

        # True: one hot encoding for categorical
        # False: no one hot encoding for sparse catecorical
        self.one_hot = True

        # define datasets
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        # define data augmentation
        self.data_augmentation = tf.keras.Sequential([
            RandomFlip("horizontal_and_vertical"),
            RandomRotation(0.4),
            RandomContrast(0.4),
            RandomTranslation(0.2, 0.2, fill_mode='reflect'),
        ])

        # define the folders of the dataset
        train_folder = 'dtd_train'
        val_folder = 'dtd_val'
        test_folder = 'dtd_test'

        # if tiled should be used
        if settings.use_tiled:
            if log: print('Using tiled dataset')
            train_folder += '_tiled'
            val_folder += '_tiled'
            test_folder += '_tiled'

        # load datasets
        self.train_ds, train_size = self.create_dataset(
            os.path.join(self.settings.dataset_path, train_folder))
        self.train_steps = math.floor(train_size / self.settings.batch_size)

        self.val_ds, val_size = self.create_dataset(
            os.path.join(self.settings.dataset_path, val_folder))
        self.val_steps = math.floor(val_size / self.settings.batch_size)

        self.test_ds, test_size = self.create_dataset(
            os.path.join(self.settings.dataset_path, test_folder),
            repeat=False)
        self.test_steps = math.floor(test_size / self.settings.batch_size)

    def _parse_function(self, image_filename, label_filename, channels: int):
        """
        Parse image and label and return them. The image is divided by 255.0 and returned as float,
        the label is returned as is in uint8 format.
        Args:
            image_filename: name of the image file
            label_filename: name of the label file
            channels: channels of the input image, (the label is always one channel)
        Returns:
            tensors for the image and label read operations
        """
        image_string = tf.io.read_file(image_filename)
        image_decoded = tf.image.decode_png(image_string, channels=channels)
        image_decoded = tf.image.convert_image_dtype(
            image_decoded, dtype=tf.float32)

        # normalize image to zero mean
        image = tf.multiply(image_decoded, 2.0)
        image = tf.subtract(image, 1.0)

        label_string = tf.io.read_file(label_filename)
        label = tf.image.decode_png(label_string, dtype=tf.uint8, channels=1)

        return image, label

    @staticmethod
    def load_files(data_dir: str):
        path = Path(data_dir)
        image_files = list(path.glob('image*.png'))
        label_files = list(path.glob('label*.png'))

        # make sure they are in the same order
        image_files.sort()
        label_files.sort()

        image_files_array = np.asarray([str(p) for p in image_files])
        label_files_array = np.asarray([str(p) for p in label_files])

        return image_files_array, label_files_array

    def create_dataset(self, data_dir: str, repeat: bool=True):
        image_files_array, label_files_array = self.load_files(data_dir)

        dataset = tf.data.Dataset.from_tensor_slices((image_files_array,
                                                      label_files_array))
        # shuffle the filename, unfortunately, then we cannot cache them
        dataset = dataset.shuffle(buffer_size=10000)
        # read the images
        dataset = dataset.map(
            lambda image, file: self._parse_function(
                image, file, self.settings.patch_channels))
        # Set the sizes of the input image, as keras needs to know them
        dataset = dataset.map(
            lambda x, y: (
                tf.reshape(x, shape=(
                    self.settings.patch_size, self.settings.patch_size, self.settings.patch_channels)),
                tf.reshape(y, shape=(
                        self.settings.patch_size, self.settings.patch_size))))
        # cut center of the label image in order to use valid filtering in the
        # network
        b = self.settings.patch_border
        if b != 0:
            dataset = dataset.map(lambda x, y:
                                    (x, y[b:-b, b:-b]))

        if self.one_hot:
            # reshape the labels to 1d array and do one-hot encoding
            dataset = dataset.map(lambda x, y:
                                  (x, tf.reshape(y, shape=[-1])))
            dataset = dataset.map(
                lambda x, y: (
                    x, tf.one_hot(
                        y, depth=self.settings.n_classes, dtype=tf.float32)))

        if self.settings.augment:
            dataset = dataset.map(
                lambda x, y: (tf.squeeze(self.data_augmentation(tf.expand_dims(x, 0), training=True), 0), y),
                num_parallel_calls=self.AUTOTUNE)

        # batch dataset
        dataset = dataset.batch(self.settings.batch_size).prefetch(1000)

        # repeat dataset
        if repeat:
            dataset = dataset.repeat()

        return dataset, image_files_array.size

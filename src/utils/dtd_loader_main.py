#
# Load the describable texture data set.
#
#
from typing import Tuple
import os
import random
import sys
import tarfile
import urllib.request
import skimage.io

import numpy as np
import scipy.io
import skimage.color


# exported consts
DTD_PATCH_WIDTH = 128
DTD_PATCH_HEIGHT = 128
DTD_LABEL_FILENAME = 'dtd_class_labels'
DTD_NUM_CLASSES = 47

DTD_URL = "http://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
DTD_DEST_ZIP_FILE = "dtd-r1.0.1.tar.gz"

# private internally used consts

# location of images and mat in the extracted directory
_IMAGE_SUB_DIR = 'dtd/images'
MAT_FILE = 'dtd/imdb/imdb.mat'

def extract_image_centers(image:np.ndarray, patch_size: Tuple):
    """Return a patch from the center of the image with the given size"""
    input_shape = image.shape
    if patch_size[0] > input_shape[0] or patch_size[1] > input_shape[1]:
        return None
    start_point = [0, 0]
    end_point = [0, 0]
    for i in [0, 1]:
        start_point[i] = (input_shape[i] - patch_size[i]) // 2
        end_point[i] = start_point[i] + patch_size[i]

    return image[start_point[0]:end_point[0], start_point[1]:end_point[1]]


def _read_mat(filename: str):
    mat = scipy.io.loadmat(filename)
    return mat


def _get_class_names_from_mat(mat):
    meta_data = mat.get('meta')
    class_names = np.array(meta_data['classes'][0][0][0])
    # make a list of that (there is probably an easier way)
    class_name_dict = {}
    for i in range(len(class_names)):
        class_name_dict[i+1] = class_names[i][0]
    return class_name_dict


def _get_images_from_mat(mat):
    image_data = mat.get('images')
    id = image_data['id'][0][0][0]
    name = image_data['name'][0][0][0]
    set = image_data['set'][0][0][0]
    class_id = image_data['class'][0][0][0]
    return id, name, set, class_id

def print_class_names(class_names):
    for i in range(len(class_names)):
        print('{0}: {1}'.format(i, class_names[i]))

def get_class_names(filename: str) -> dict:
    mat = _read_mat(filename)
    return _get_class_names_from_mat(mat)

def get_data_sets(filename: str):
    """Get the data set description from the file"""
    train_id = 1
    val_id = 2
    test_id = 3
    train = []
    val = []
    test = []
    mat = _read_mat(filename)
    id, name, set, class_id = _get_images_from_mat(mat)
    for i in range(len(id)):
        data = {}
        data['id'] = id[i]
        data['set'] = set[i]
        data['file_name'] = name[i][0]
        data['class_id'] = class_id[i]

        if set[i] == train_id:
            train.append(data)
        elif set[i] == val_id:
            val.append(data)
        elif set[i] == test_id:
            test.append(data)
    return train, val, test


def convert_data(data_set, destination_dir, dataset_dir, class_ids=None):
    """Convert the data sets. If class_ids is None then all classes will
    be converted
    Args:
        data_set: the dataset to convert
        patch_file_basename: basename of the generated patch file, numbers and extension will be appended
        dataset_dir: directory where the downloaded data was extracted
        class_ids: the class ids that should be converted
    """
    image_id = 0
    if class_ids is None:
        class_ids = range(DTD_NUM_CLASSES)

    # generate label images for all the needed class ids
    labels = dict()
    for class_id in class_ids:
        label = np.full((DTD_PATCH_WIDTH, DTD_PATCH_HEIGHT), class_id, np.uint8)
        labels[class_id] = label

    for i in range(len(data_set)):
        class_id = data_set[i]['class_id']
        if class_id in class_ids:
            image_directory = os.path.join(dataset_dir, _IMAGE_SUB_DIR)
            filename = os.path.join(image_directory, data_set[i]['file_name'])

            image = skimage.io.imread(filename)
            patch = extract_image_centers(image, (DTD_PATCH_WIDTH, DTD_PATCH_HEIGHT))

            # convert image to 8bit grayscale
            patch_grey = skimage.img_as_ubyte(skimage.color.rgb2gray(patch))
            patch_grey = np.reshape(patch_grey, (DTD_PATCH_WIDTH, DTD_PATCH_HEIGHT))

            # save patch and label image in directory
            image_name = 'image_{:05}.png'.format(image_id)
            label_name = 'label_{:05}.png'.format(image_id)
            filename = os.path.join(destination_dir, image_name)
            # save the image
            skimage.io.imsave(filename, patch_grey)

            filename = os.path.join(destination_dir, label_name)
            # save the label
            skimage.io.imsave(filename, labels[class_id])
            image_id += 1

    return image_id


def convert_classes(class_ids=None, dataset_dir='dtd', dest_dir_root='.'):

    filename_for_desc = os.path.join(dataset_dir, MAT_FILE)
    if not os.path.exists(dest_dir_root):
        os.makedirs(dest_dir_root)

    train, val, test = get_data_sets(filename_for_desc)

    print("Converting dtd data set for classes: {}".format(class_ids))

    print("Converting validation data set")
    random.shuffle(val)
    dest_dir = os.path.join(dest_dir_root, 'dtd_val')
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    nr_images_val = convert_data(val, dest_dir, dataset_dir, class_ids)
    print("Number of images: {}".format(nr_images_val))

    print("Converting train data set")
    random.shuffle(train)
    dest_dir = os.path.join(dest_dir_root, 'dtd_train')
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    nr_images_train = convert_data(train, dest_dir, dataset_dir, class_ids)
    print("Number of images: {}".format(nr_images_train))

    print("Converting test data set")
    random.shuffle(test)
    dest_dir = os.path.join(dest_dir_root, 'dtd_test')
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    nr_images_test = convert_data(test, dest_dir, dataset_dir, class_ids)
    print("Number of images: {}".format(nr_images_test))


def download_dataset(dataset_dir='.'):
    """Downloads DTD data set.

    Args:
      dataset_dir: The directory where downloaded files are stored.
    """
    filepath = os.path.join(dataset_dir, DTD_DEST_ZIP_FILE)

    if not os.path.exists(filepath):
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        print('Downloading file %s...' % DTD_DEST_ZIP_FILE)

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %.1f%%' % (
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DTD_URL,
                                                 filepath,
                                                 _progress)
        print()
        size = os.path.getsize(filepath)
        print('Successfully downloaded', DTD_DEST_ZIP_FILE, size, 'bytes.')
    else:
        print('File already downloaded')

def extract_dataset(dataset_dir = '.'):
    # check if one of the extracted directories already exists, if yes, do not extract
    image_dir = os.path.join(dataset_dir, _IMAGE_SUB_DIR)
    if not os.path.exists(image_dir):
        filepath = os.path.join(dataset_dir, DTD_DEST_ZIP_FILE)
        tarfile.open(filepath, 'r:gz').extractall(dataset_dir)
        print('Successfully extracted files')
    else:
        print('Image directory exists: not extracting files')


def download_and_convert(dataset_dir = '.', dest_dir_root='.'):
    """Download the DTD file, unpack it and convert the data set."""
    # first check the resulting directories, if they exist, do not do anything
    train_dir = os.path.join(dest_dir_root, 'dtd_train')
    val_dir = os.path.join(dest_dir_root, 'dtd_val')
    test_dir = os.path.join(dest_dir_root, 'dtd_test')

    if os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir):
        print('Directories dtd_train, dtd_val and dtd_test already exist at target location')
        print('Remove the directories to download the data again')
        return

    download_dataset(dataset_dir)
    extract_dataset(dataset_dir)

    class_ids = list(range(DTD_NUM_CLASSES))
    convert_classes(class_ids, dataset_dir, dest_dir_root)


if __name__ == '__main__':
    download_and_convert()

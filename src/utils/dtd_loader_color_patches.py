#
# Load the describable texture data set.
#
# 29.4.2019: Added support for color patches
#
from typing import Tuple, List
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


def _calculate_tiles(input_shape: Tuple, patch_size: Tuple, patch_stride=None) \
        -> List:
    """Calculate the starting points of all tiles"""
    if patch_stride is None:
        patch_stride = (patch_size[0], patch_size[1])

    if patch_stride[0] < 1 or patch_stride[1] < 1:
        raise ValueError('Strides must be larger than 0')

    if patch_size[0] > input_shape[0] or patch_size[1] > input_shape[1]:
        return None

    # result is a list of points (tuples)
    result = []

    # number of images that fit in each dimension
    nr_images = [1, 1]
    for i in [0, 1]:
        nr_images[i] += (input_shape[i] - patch_size[i]) // patch_stride[i]

    # starting coordinates for the tiles and total length of tiled area,
    # the tiled area is centered
    start_coord = [0, 0]
    tiled_area_length = [0, 0]

    for i in [0, 1]:
        tiled_area_length[i] = patch_size[i] + (nr_images[i]-1)*patch_stride[i]
        start_coord[i] = (input_shape[i] - tiled_area_length[i]) // 2

    for i in range(nr_images[0]):
        for j in range(nr_images[1]):
            x = start_coord[0] + i * patch_stride[0]
            y = start_coord[1] + j * patch_stride[1]
            result.append((x, y))
    return result


def tile_image(image: np.ndarray, patch_size: Tuple, patch_stride=None) -> Tuple:
    """
    Generate patches of size patch_size from the image that are patch_stride
    pixels apart. The patches are centered. Returns the patches in a list, and the start points
    of the patches in another
    Args:
        image: The image to tile
        patch_size: tuple of 2 int containing the patch width and height
        patch_stride: tuple of 2 int containing the strides, if None, the patch_size will be used as stride

    Returns:
        tuple of a list of the patches and a list of the start points of each patch
    """
    points = _calculate_tiles(image.shape, patch_size, patch_stride)

    # return a list of patches
    patches = []

    for i in range(len(points)):
        start_point = points[i]
        end_point = (start_point[0] + patch_size[0], start_point[1] + patch_size[1])
        patch = image[start_point[0]:end_point[0], start_point[1]:end_point[1]]
        patches.append(patch)

    return patches, points


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
        destination_dir: directory where to write the files
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
            skimage.io.imsave(filename, patch_grey, check_contrast=False)

            filename = os.path.join(destination_dir, label_name)
            # save the label
            skimage.io.imsave(filename, labels[class_id], check_contrast=False)
            image_id += 1

    return image_id


def convert_data_tiled(data_set, destination_dir, dataset_dir,
                       patch_size: Tuple, patch_stride=None,
                       class_ids=None):
    """Convert the data sets. If class_ids is None then all classes will
    be converted. Used image tiles to convert the original images. Images from the
    same original image will end up in consecutively numbered files, so they should
    be shuffled when used
    Args:
        data_set: the dataset to convert
        destination_dir: directory where to write the files
        dataset_dir: directory where the downloaded data was extracted
        patch_size: size of patches
        patch_stride: stride between patches, or none if patch size should be used
        class_ids: the class ids that should be converted
    """
    image_id = 0
    if class_ids is None:
        class_ids = range(DTD_NUM_CLASSES)

    # generate label images for all the needed class ids
    labels = dict()
    for class_id in class_ids:
        label = np.full((patch_size[0], patch_size[1]), class_id, np.uint8)
        labels[class_id] = label

    for i in range(len(data_set)):
        class_id = data_set[i]['class_id']
        if class_id in class_ids:
            image_directory = os.path.join(dataset_dir, _IMAGE_SUB_DIR)
            filename = os.path.join(image_directory, data_set[i]['file_name'])

            image = skimage.io.imread(filename)
            patches, _ = tile_image(image, patch_size, patch_stride)

            for patch in patches:
                patch_ubyte = skimage.img_as_ubyte(patch)

                # save patch and label image in directory
                image_name = 'image_{:05}.png'.format(image_id)
                label_name = 'label_{:05}.png'.format(image_id)
                filename = os.path.join(destination_dir, image_name)
                # save the image
                skimage.io.imsave(filename, patch_ubyte, check_contrast=False)

                filename = os.path.join(destination_dir, label_name)
                # save the label
                skimage.io.imsave(filename, labels[class_id], check_contrast=False)
                image_id += 1

    return image_id


def convert_classes(class_ids,
                    dataset_dir,
                    train_dir, val_dir, test_dir, tiled=False):

    filename_for_desc = os.path.join(dataset_dir, MAT_FILE)

    train, val, test = get_data_sets(filename_for_desc)

    print("Converting dtd data set for classes: {}".format(class_ids))

    print("Converting validation data set")
    random.shuffle(val)
    dest_dir = val_dir
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if tiled:
        nr_images_val = convert_data_tiled(val, dest_dir, dataset_dir,
                                           patch_size=(DTD_PATCH_WIDTH, DTD_PATCH_HEIGHT),
                                           patch_stride=None,
                                           class_ids=class_ids)
    else:

        nr_images_val = convert_data(val, dest_dir, dataset_dir, class_ids)
    print("Number of images: {}".format(nr_images_val))

    print("Converting train data set")
    random.shuffle(train)
    dest_dir = train_dir
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if tiled:
        nr_images_train = convert_data_tiled(train, dest_dir, dataset_dir,
                                             patch_size=(DTD_PATCH_WIDTH, DTD_PATCH_HEIGHT),
                                             patch_stride=None,
                                             class_ids=class_ids)
    else:
        nr_images_train = convert_data(train, dest_dir, dataset_dir, class_ids)
    print("Number of images: {}".format(nr_images_train))

    print("Converting test data set")
    random.shuffle(test)
    dest_dir = test_dir
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if tiled:
        nr_images_test = convert_data_tiled(test, dest_dir, dataset_dir,
                                            patch_size=(DTD_PATCH_WIDTH, DTD_PATCH_HEIGHT),
                                            patch_stride=None,
                                            class_ids=class_ids)
    else:
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


def download_and_convert(dataset_dir = '.', dest_dir_root='.', tiled=False):
    """Download the DTD file, unpack it and convert the data set."""
    # first check the resulting directories, if they exist, do not do anything
    if tiled:
        train_dir = os.path.join(dest_dir_root, 'dtd_train_tiled')
        val_dir = os.path.join(dest_dir_root, 'dtd_val_tiled')
        test_dir = os.path.join(dest_dir_root, 'dtd_test_tiled')
    else:
        train_dir = os.path.join(dest_dir_root, 'dtd_train')
        val_dir = os.path.join(dest_dir_root, 'dtd_val')
        test_dir = os.path.join(dest_dir_root, 'dtd_test')

    if os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir):
        print('Directories for train, val and test already exist at target location')
        print('Remove the directories to convert the data again')
        return

    download_dataset(dataset_dir)
    extract_dataset(dataset_dir)

    class_ids = list(range(DTD_NUM_CLASSES))
    convert_classes(class_ids, dataset_dir, train_dir, val_dir, test_dir, tiled)


if __name__ == '__main__':
    download_and_convert()

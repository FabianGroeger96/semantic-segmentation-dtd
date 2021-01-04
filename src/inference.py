import os
import random
import pylab
import skimage.io

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.settings.settings import Settings
from src.dataset.dtd_dataset import DTDDataset
from src.utils.utils import bcolors


# generate a labeled color image
def get_color_representation(data):
    side_length = len(data)
    result = np.zeros((side_length, side_length, 3))
    for col in range(side_length):  # y
        for row in range(side_length):  # x
            class_for_pixel = data[row, col]
            # rgb
            result[row, col, 0] = colors[class_for_pixel][0]  # r
            result[row, col, 1] = colors[class_for_pixel][1]  # g
            result[row, col, 2] = colors[class_for_pixel][2]  # b

    return result


if __name__ == '__main__':
    # global settings
    settings = Settings()

    # model path
    model_path = 'experiments/DTD/ResNeSt-0.001-0.4-02012021-141555/saved_model/'

    # create a distinct color map for verification
    cm = pylab.get_cmap('gist_rainbow')
    colors = []  # list of (r,g,b,a)
    for i in range(settings.n_classes):
        colors.append(cm(1.*i/settings.n_classes))

    # create list for label names
    label_names = settings.labels.split(',')
    # insert label at idx 0, since labels start with 1
    label_names.insert(0, 'Not defined')

    # read files
    val_name = 'dtd_val'
    if settings.use_tiled: val_name += '_tiled'
    val_path = os.path.join(settings.dataset_path, val_name)
    image_files_array, label_files_array = DTDDataset.load_files(val_path)

    # get a random sample from the dataset
    rand_idx = random.sample(range(len(image_files_array)), 1)
    rand_img_path = image_files_array[rand_idx][0]
    rand_lbl_path = label_files_array[rand_idx][0]

    # load the random image
    rand_img = skimage.io.imread(rand_img_path).astype(np.float32)
    rand_img /= (255.0 / 2) - 1

    # load the corresponding label
    rand_lbl = skimage.io.imread(rand_lbl_path)

    # load the model
    model = tf.keras.models.load_model(model_path)

    # evaluate on test set if needed
    if True:
        # create dataset
        dataset = DTDDataset.get_instance(settings=settings,
                                        log=settings.log)
        # evaluate
        model.evaluate(dataset.test_ds)

    # predict the random sample
    prediction = model.predict(tf.expand_dims(rand_img, axis=0))
    highest_probability_map = np.reshape(prediction, (128, 128, 47))
    highest_probability_map = np.argmax(highest_probability_map, axis=2)
    unique, counts = np.unique(highest_probability_map, return_counts=True)
    occurence_dict = dict(zip(unique, counts))

    # plot the result
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
    axs[0].set(title="Original")
    axs[0].imshow(rand_img, cmap='gray')

    axs[1].set(title=f"prediction")
    axs[1].imshow(get_color_representation(highest_probability_map))

    label_value = rand_lbl[0][0]
    label_name = label_names[label_value]
    axs[2].set(title=f"{label_name}")
    axs[2].imshow(get_color_representation(rand_lbl))

    # transform the occurence dict, so that the key is the name of label rather
    # then the index
    occurence_dict_lbls = {'{0} ({1})'.format(label_names[key], str(key)): value for (key, value)
                           in occurence_dict.items()}

    # print summary of prediction
    print(bcolors.OKGREEN + 'name of random img:' + bcolors.ENDC,
          rand_lbl_path)
    print(bcolors.OKGREEN + 'correct label:' + bcolors.ENDC,
          label_value,
          '(' + label_names[label_value] + ')')
    print(bcolors.OKGREEN + 'how many predicted correct:' + bcolors.ENDC,
          occurence_dict.get(label_value, 0))
    print(bcolors.OKGREEN + 'predicted occurences:' + bcolors.ENDC,
          occurence_dict_lbls)

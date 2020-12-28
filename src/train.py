import numpy as np
import tensorflow as tf

from src.dataset.dtd_dataset import DTDDataset

if __name__ == '__main__':
    # create dataset
    data_path = 'data'
    dataset = DTDDataset.get_instance(data_path)

    pass

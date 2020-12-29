from enum import Enum


class Settings:

    def __init__(self):
        # global configs
        self.epochs = 50
        self.batch_size = 16

        # model config
        self.model = Models.U_NET
        self.layer_depth = 3
        self.filters_root = 64
        self.dropout_rate = 0.2

        # dataset config
        self.dataset_path = 'data'
        self.n_classes = 47
        self.patch_size = 128
        self.patch_channels = 1  # 1: gray, 3: color
        self.patch_border = 20
        self.augment = True

class Models(Enum):
    SIMPLE_FCN = 1
    U_NET = 2

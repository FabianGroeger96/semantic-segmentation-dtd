from enum import Enum


class Settings:

    def __init__(self):
        # global configs
        self.epochs = 50
        self.batch_size = 1

        # model config
        self.model = Models.RESNEST
        self.layer_depth = 3
        self.filters_root = 64
        self.dropout_rate = 0.2

        # dataset config
        self.dataset_path = 'data'
        self.n_classes = 47
        self.patch_size = 128
        self.patch_channels = 1  # 1: gray, 3: color
        self.patch_border = 0  # unet: 20, fcn: 3, resnest: 0
        self.augment = True

class Models(Enum):
    SIMPLE_FCN = 1
    U_NET = 2
    RESNEST = 3

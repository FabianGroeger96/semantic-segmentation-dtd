from enum import Enum


class Settings:

    def __init__(self):
        # global configs
        self.epochs = 300
        self.batch_size = 8
        self.log = True

        # model config
        self.model = Models.RESNEST
        self.dropout_rate = 0.4
        # U-Net
        self.layer_depth = 3
        self.filters_root = 64

        # dataset config
        self.dataset_path = 'data'
        self.n_classes = 47
        self.labels = 'banded,blotchy,braided,bubbly,bumpy,chequered,\
            cobwebbed,cracked,crosshatched,crystalline,dotted,fibrous,\
            flecked,freckled,frilly,gauzy,grid,grooved,honeycombed,\
            interlaced,knitted,lacelike,lined,marbled,matted,meshed,\
            paisley,perforated,pitted,pleated,polka-dotted,porous,\
            potholed,scaly,smeared,spiralled,sprinkled,stained,\
            stratified,striped,studded,swirly,veined,waffled,woven,\
            wrinkled,zigzagged'
        self.patch_size = 128
        self.patch_channels = 3  # 1: gray, 3: color
        self.patch_border = 0  # unet: 20, fcn: 3, resnest: 0
        self.use_tiled = True
        self.augment = True


class Models(Enum):
    SIMPLE_FCN = 1
    U_NET = 2
    RESNEST = 3
    RESNET = 4

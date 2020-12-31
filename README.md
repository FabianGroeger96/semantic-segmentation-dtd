# Semantic Segmentation Project
Semantic Segmentation for the Describable Textures Dataset (DTD)

## Download DTD Dataset
To download the DTD Dataset, you can use the utility files `dtd_loader_color_patches.py` and `dtd_loader_main.py`, patches will split the images in multiple patches and main will download the entire image.

Both can be called using, within the files the data directory can be specified, for the location of the dataset:   
`python -m src.utils.dtd_loader_main`

## Training
To train a model, you can run:   
`python -m src.train`

All of the configurations for the training can be done in the `settings.py` script.

Currently these models are implemented:
- Simple FCN (3 convs)
- U-Net
- ResNet (with transposed convs as decoder)
- [ResNeSt](https://arxiv.org/abs/2004.08955) (with transposed convs as decoder)

## Inference
To run inference on a model, which will show the segmentation mask of a random sample in the validation set, you can run:   
`python -m src.inference`

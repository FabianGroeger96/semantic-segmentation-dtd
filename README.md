# Semantic Segmentation Project
Semantic Segmentation for the Describable Textures Dataset (DTD)

## Download DTD Dataset
To download the DTD Dataset, you can use the utility files `dtd_loader_color_patches.py` and `dtd_loader_main.py`, patches will split the images in multiple patches and main will download the entire image.

Both can be called using, within the files the data directory can be specified, for the location of the dataset:
`python -m src.utils.dtd_loader_main`

import tensorflow as tf

from src.dataset.dtd_dataset import DTDDataset
from src.models.fcn.simple_fcn import SimpleFCN
from src.models.u_net.u_net_model import UNet
from src.settings.settings import Settings, Models
from src.utils.utils import create_experiment_folders
from src.models.resnest.resnest import ResNest


if __name__ == '__main__':
    # global settings
    settings = Settings()

    # create dataset
    dataset = DTDDataset.get_instance(settings)

    # create and build the model
    if settings.model is Models.SIMPLE_FCN:
        # Simple FCN
        model = SimpleFCN(settings.n_classes,
                          settings.patch_size,
                          settings.patch_border,
                          settings.patch_channels,
                          dropout_rate=settings.dropout_rate)

    elif settings.model is Models.U_NET:
        # U-Net
        model = UNet(num_classes=settings.n_classes,
                     img_size=settings.patch_size,
                     img_border=settings.patch_border,
                     nr_channels=settings.patch_channels,
                     layer_depth=settings.layer_depth,
                     filters_root=settings.filters_root,
                     dropout_rate=settings.dropout_rate)
    elif settings.model is Models.RESNEST:
        # ResNeSt
        input_shape = [settings.patch_size,
                       settings.patch_size,
                       settings.patch_channels]
        model = ResNest(
            verbose=True,
            input_shape=input_shape,
            n_classes=settings.n_classes,
            dropout_rate=settings.dropout_rate,
            blocks_set=[2, 2, 2, 2],  # ResNeSt18: [2, 2, 2, 2], ResNeSt50: [3, 4, 6, 3]
            stem_width=32,
            radix=2,
            groups=1,
            bottleneck_width=64,
            deep_stem=True,
            avg_down=True,
            avd=True,
            avd_first=False,
            using_cb=True).build()
        model.model_name = 'ResNeSt'

    # build the model
    in_shape = [1,
                settings.patch_size,
                settings.patch_size,
                settings.patch_channels]
    model.build(in_shape)
    print(model.summary())

    # create the paths for the experiment
    paths = create_experiment_folders(dataset.name,
                                      model.model_name,
                                      post_fix='lr1e2')

    # define the loss function
    loss = tf.keras.losses.categorical_crossentropy
    # define the optimizer
    optimizer = tf.keras.optimizers.Adam(lr=1e-2)
    # define the metrics to track and visualize in tensorboard
    metrics = ['categorical_crossentropy',
               'categorical_accuracy']

    # compile the model
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    # define the callbacks
    callbacks = [tf.keras.callbacks.ModelCheckpoint(paths['save'],
                                                    save_best_only=True),
                 tf.keras.callbacks.TensorBoard(log_dir=paths['tensorboard'],
                                                update_freq=1,
                                                histogram_freq=1,
                                                profile_batch=0,
                                                write_graph=True)]
    # train the model
    model.fit(dataset.train_ds,
              validation_data=dataset.val_ds,
              validation_steps=dataset.val_steps,
              steps_per_epoch=dataset.train_steps,
              epochs=settings.epochs,
              callbacks=callbacks)

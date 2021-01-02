import os
import tensorflow as tf

from src.dataset.dtd_dataset import DTDDataset
from src.models.fcn.simple_fcn import SimpleFCN
from src.models.u_net.u_net_model import UNet
from src.models.resnest.resnest import ResNest
from src.models.resnet.resnet import ResNet18
from src.settings.settings import Settings, Models
from src.utils.utils import create_experiment_folders


def check_if_GPUs():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been
            print(e)

def scheduler(epoch, lr):
    if epoch < 50:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


if __name__ == '__main__':
    # check if GPU
    check_if_GPUs()

    # global settings
    settings = Settings()

    # create dataset
    dataset = DTDDataset.get_instance(settings=settings,
                                      log=settings.log)

    # create and build the model
    if settings.model is Models.SIMPLE_FCN:
        # Simple FCN
        lr = 1e-4
        model = SimpleFCN(settings.n_classes,
                          settings.patch_size,
                          settings.patch_border,
                          settings.patch_channels,
                          dropout_rate=settings.dropout_rate)

    elif settings.model is Models.U_NET:
        # U-Net
        lr = 1e-4
        model = UNet(num_classes=settings.n_classes,
                     img_size=settings.patch_size,
                     img_border=settings.patch_border,
                     nr_channels=settings.patch_channels,
                     layer_depth=settings.layer_depth,
                     filters_root=settings.filters_root,
                     dropout_rate=settings.dropout_rate)

    elif settings.model is Models.RESNEST:
        # ResNeSt
        lr = 1e-2
        input_shape = [settings.patch_size,
                       settings.patch_size,
                       settings.patch_channels]
        model = ResNest(
            verbose=settings.log,
            input_shape=input_shape,
            n_classes=settings.n_classes,
            dropout_rate=settings.dropout_rate,
            # ResNeSt18: [2, 2, 2, 2], ResNeSt50: [3, 4, 6, 3]
            blocks_set=[2, 2, 2, 2],
            stem_width=32,
            radix=2,
            groups=1,
            bottleneck_width=64,
            deep_stem=False,
            avg_down=False,
            avd=True,
            avd_first=False,
            using_cb=False).build()
        model.model_name = 'ResNeSt'

    elif settings.model is Models.RESNET:
        # ResNet
        lr = 1e-3
        model = ResNet18()

    # build the model
    in_shape = [1,
                settings.patch_size,
                settings.patch_size,
                settings.patch_channels]
    model.build(in_shape)
    model.summary()

    # create the paths for the experiment
    paths = create_experiment_folders(dataset.name,
                                      model.model_name,
                                      post_fix='{0}-{1}'.format(str(lr),
                                                                str(settings.dropout_rate)))

    # define the loss function
    loss = tf.keras.losses.categorical_crossentropy
    # define the optimizer
    optimizer = tf.keras.optimizers.Adam(lr=lr)
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
                 tf.keras.callbacks.CSVLogger(os.path.join(paths['log'],
                                                           'training.log')),
                 tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
                 tf.keras.callbacks.TensorBoard(log_dir=paths['tensorboard'],
                                                update_freq=1,
                                                histogram_freq=10,
                                                profile_batch=0,
                                                write_graph=True)]
    # train the model
    model.fit(dataset.train_ds,
              validation_data=dataset.val_ds,
              validation_steps=dataset.val_steps,
              steps_per_epoch=dataset.train_steps,
              epochs=settings.epochs,
              callbacks=callbacks)

    # evaluate the model on the test set
    model.evaluate(dataset.test_ds,
                   callbacks=callbacks)

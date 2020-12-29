import tensorflow as tf

from src.dataset.dtd_dataset import DTDDataset
from src.models.simple_fcn import SimpleFCN
from src.settings.settings import Settings
from src.utils.utils import create_experiment_folders

if __name__ == '__main__':
    # global settings
    settings = Settings()
    # create dataset
    data_path = 'data'
    dataset = DTDDataset.get_instance(data_path, settings=settings)

    # create and build the model
    model = SimpleFCN(settings.n_classes,
                      settings.patch_size,
                      settings.patch_border,
                      settings.patch_channels,
                      dropout_rate=0.3)
    in_shape = [settings.batch_size,
                settings.patch_size,
                settings.patch_size,
                settings.patch_channels]
    model.build(in_shape)
    print(model.summary())

    # create the paths for the experiment
    paths = create_experiment_folders(dataset.name,
                                      model.model_name)

    # compile the model
    loss = tf.keras.losses.categorical_crossentropy
    optimizer = tf.keras.optimizers.Adam(lr=1e-2)
    metrics = ['categorical_crossentropy',
               'categorical_accuracy']
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    # define the callbacks
    callbacks = [tf.keras.callbacks.ModelCheckpoint(paths['save'],
                                                    save_best_only=True),
                 tf.keras.callbacks.TensorBoard(log_dir=paths['tensorboard'],
                                                update_freq=1,
                                                histogram_freq=1,
                                                profile_batch=50,
                                                write_graph=True)]
    # train the model
    model.fit(dataset.train_ds,
              validation_data=dataset.val_ds,
              validation_steps=dataset.val_steps,
              steps_per_epoch=dataset.train_steps,
              epochs=50,
              callbacks=callbacks)

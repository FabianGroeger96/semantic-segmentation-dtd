import tensorflow as tf

from src.dataset.dtd_dataset import DTDDataset
from src.models.simple_fcn import SimpleFCN
from src.utils.utils import create_experiment_folders

if __name__ == '__main__':
    # create dataset
    data_path = 'data'
    dataset = DTDDataset.get_instance(data_path)

    # create and build the model
    model = SimpleFCN(dataset.n_classes,
                      dataset.patch_channels,
                      dropout_rate=0.3)
    in_shape = [dataset.batch_size,
                dataset.patch_size,
                dataset.patch_size,
                dataset.patch_channels]
    model.build(in_shape)
    print(model.summary())

    # create the paths for the experiment
    paths = create_experiment_folders(dataset.name,
                                      model.model_name)

    # compile the model
    loss = tf.keras.losses.categorical_crossentropy
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    metrics = ['categorical_crossentropy',
               'categorical_accuracy']
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    # train the model
    callbacks = [tf.keras.callbacks.ModelCheckpoint(paths['save'],
                                                    save_best_only=True),
                 tf.keras.callbacks.TensorBoard(log_dir=paths['tensorboard'],
                                                update_freq=1,
                                                histogram_freq=1,
                                                write_graph=True)]
    model.fit(dataset.train_ds,
              validation_data=dataset.val_ds,
              validation_steps=dataset.val_steps,
              steps_per_epoch=dataset.train_steps,
              epochs=50,
              callbacks=callbacks)

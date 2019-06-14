"""Main module for Temporal training"""
#pylint: disable=superfluous-parens
import os
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from Temporal.Networks.network_temporal import load_network as load_network
from Temporal.batch_generator import BatchGenerator
from Temporal.data_configuration import Config
from Temporal.stop_training_on_request import StopTrainingOnInput

from Misc.misc import save_results
from Misc.misc import create_new_folder

class Trainer(object):
    """ Main class for training a new model """
    def __init__(self):
        self.conf = Config()
        self.history = None
        self.train_generator = None
        self.validation_generator = None
        self.network_handler = None
        self.folder = None
        self.create_results_folder()
        self.checkpoint_path_loss = "Stored_models/" + \
            str(self.folder) + "/Checkpoints/train_loss-{epoch:02d}-{loss:.3f}.hdf5"
        self.checkpoint_path_val_loss = "Stored_models/" + \
            str(self.folder) + "/Checkpoints/val_loss-{epoch:02d}-{val_loss:.3f}.hdf5"
        self.logs_dir = "./logs"

    def create_results_folder(self):
        """ Creates two folders:
        1. The folder that contains all results - defined by an integer
        2. The Checkoint folder used to store all checkpoints
        """
        self.folder = create_new_folder("../Temporal/Stored_models/")
        try:
            os.mkdir("Stored_models/" + str(self.folder) + "/Checkpoints")
            print("Created checpoints folder for " + str(self.folder))
        except OSError:
            print("Creation of the directory %s failed" % "Checkpoints")

    def initialise_generator_and_net(self):
        """ Creates a batch generator and a network handler"""
        self.train_generator = BatchGenerator(self.conf)
        self.validation_generator = BatchGenerator(self.conf, data="Validation_data")
        self.network_handler = load_network(self.conf)

    def train(self):
        """ Trains the model """
        self.network_handler.model.compile(
            loss=self.conf.loss_functions,
            optimizer=self.conf.train_conf.optimizer,
        )
        self.history = self.network_handler.model.fit_generator(
            self.train_generator,
            validation_data=self.validation_generator, #[self.validation_data_X, self.validation_data_Y],
            steps_per_epoch=len(self.train_generator),
            validation_steps=len(self.validation_generator),
            epochs=self.conf.train_conf.epochs,
            callbacks=[
                ModelCheckpoint(self.checkpoint_path_loss, monitor='loss', save_best_only=True, period=int(np.floor(self.conf.train_conf.epochs/10))),
                ModelCheckpoint(self.checkpoint_path_val_loss, monitor='val_loss', save_best_only=True),
                EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto', baseline=None, restore_best_weights=True),
                #TensorBoard(self.validation_generator2, int(len(self.validation_generator2)/16), 16, log_dir="{}/{}".format(self.logs_dir, time.time()), histogram_freq=1, batch_size=16)
                TensorBoard(histogram_freq=0, batch_size=16, write_images=False, write_grads=False),
                StopTrainingOnInput()
            ],
            use_multiprocessing=True,
            workers=12,
            verbose=1
        )

    def save(self):
        """ Saves Model(object, image and description), History and config """
        save_results(self, path="../Temporal/")

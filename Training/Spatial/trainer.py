"""Main module for spatial training"""

from Spatial.network import load_network
from Spatial.batch_generator import BatchGenerator
from Spatial.data_configuration import Config
from Misc.misc import save_results

class Trainer(object):
    """ Main class for training a new model """
    def __init__(self):
        self.conf = Config()
        self.history = None
        self.update_config()
        self.generator = BatchGenerator(self.conf)
        self.network_handler = load_network(self.conf)
        print(self.network_handler.model.summary())


    def update_config(self):
        """use this to update configuration when training several models"""
        pass

    def train(self):
        """ Trains the model """
        self.network_handler.model.compile(
            loss=self.conf.train_conf.loss,
            optimizer=self.conf.train_conf.optimizer,
            metrics=self.conf.train_conf.metrics
        )        
        self.history = self.network_handler.model.fit_generator(
            self.generator.generate(),
            steps_per_epoch=self.generator.get_number_of_steps_per_epoch(),
            epochs=self.conf.train_conf.epochs,
        )

    def save(self):
        """ Saves Model(object, image and description), History and config """
        save_results(
            self.network_handler,
            self.history,
            self.conf,
            path="../Spatial/",
            folder=None
        )

TRAINER = Trainer()
TRAINER.train()
TRAINER.save()

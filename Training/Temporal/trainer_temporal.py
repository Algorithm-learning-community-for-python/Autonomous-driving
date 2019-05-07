from Temporal.network_temporal import load_network
from Temporal.sequence_data_creater import SequenceCreator
from Temporal.batch_generator import BatchGenerator
from Misc.data_configuration import Config
from Misc.misc import save_results

class Trainer(object):
    """ Main class for training a new model """
    def __init__(self):
        self.conf = Config()

        self.network_handler = None
        self.history = None
        self.generator = BatchGenerator()

        self.update_config()


        create_sequence_data = False
        if create_sequence_data:
            SequenceCreator(self.conf)


        # Init network handler and its model
        self.network_handler = load_network(self.conf)

    def update_config(self):
        """use this to update configuration when training several models"""
        pass

    def train(self):
        """ Trains the model """
        self.network_handler.model.compile(
            loss=self.conf.train_conf.loss,
            optimizer=self.conf.train_conf.optimizer,
            #metrics=self.conf.train_conf.metrics
        )
        self.history = self.network_handler.model.fit_generator(
            self.generator.generate(),
            steps_per_epoch=self.generator.get_number_of_steps_per_epoch(),
            epochs=self.conf.train_conf.epochs,
        )
        self.network_handler.model.save("Current_model/model.h5")

    def save(self):
        """ Saves Model(object, image and description), History and config """
        save_results(
            self.network_handler,
            self.history,
            self.conf,
            path="../Temporal/",
            folder=None
        )

TRAINER = Trainer()
TRAINER.train()
TRAINER.save()

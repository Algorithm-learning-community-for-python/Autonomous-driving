import os
from keras.utils import plot_model

from data_handler_temporal import DataHandler
from network_temporal import load_network
from visualizer import Visualizer
from data_configuration import Config
from sequence_data_creater import SequenceCreator
from batch_generator import BatchGenerator

class Trainer:
    def __init__(self):
        self.conf = Config()

        self.network_handler = None
        self.history = None
        self.model = None
        self.generator = BatchGenerator()
        #self.visualizer = Visualizer()

        create_sequence_data = False
        if create_sequence_data:
            SequenceCreator()

        self.init_network()

    def init_network(self):
        self.network_handler = load_network(self.conf.input_size_data, self.conf.input_data)
        plot_model(self.network_handler.model, to_file="model.png")
        print(self.network_handler.model.summary())
        with open("model.txt", "w") as fh:
            self.network_handler.model.summary(print_fn=lambda x: fh.write(x + "\n"))
        fh.close()
        print(self.get_model_memory_usage(self.conf.train_conf.batch_size, self.network_handler.model))
        raw_input()

    def get_model_memory_usage(self, batch_size, model):
        import numpy as np
        from keras import backend as K

        shapes_mem_count = 0
        for l in model.layers:
            single_layer_mem = 1
            for s in l.output_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
        non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

        number_size = 4.0
        if K.floatx() == 'float16':
                number_size = 2.0
        if K.floatx() == 'float64':
                number_size = 8.0

        total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
        gbytes = np.round(total_memory / (1024.0 ** 3), 3)
        return gbytes


    def train(self):
        self.model = self.network_handler.model

        self.model.compile(
            loss=self.conf.train_conf.loss,
            optimizer=self.conf.train_conf.optimizer,
            #metrics=self.conf.train_conf.metrics
        )

        self.history = self.model.fit_generator(
            self.generator.generate(),
            steps_per_epoch = self.generator.get_number_of_steps_per_epoch(),
            epochs=self.conf.train_conf.epochs,
            max_queue_size=1,
            workers = 1,
            pickle_safe=False
        )

        self.model.save('model.h5')

    def store_results(self, folder=None):
        if folder:
            folder = folder
        else:
            last_folder = 0
            for folder in os.listdir('Stored_models'):
                if int(folder) >= last_folder:
                    last_folder = int(folder)+1
            folder = last_folder 

        path = "Stored_models/" + str(folder)

        try:  
            os.mkdir(path)

        except OSError:  
            print ("Creation of the directory %s failed" % path)
        else:  
            print ("Successfully created the directory %s " % path)

        # Store model
        self.model.save(path + "/model.h5")

        # Store image of model
        plot_model(self.network_handler.model, to_file=path + '/model.png')

        f = open(path + "/conf.txt", "wb+")

        # Store config
        f.write(str(self.conf.train_conf.__dict__) + "\n")

        # Store settings
        f.write(str(self.conf.input_data) + "\n")
        f.write(str(self.conf.input_size_data) + "\n")

        # Store training history
        f.write(str(self.history.history) + "\n")
        f.close()


trainer = Trainer()
#trainer.data_handler.plot_data()
trainer.train()
# trainer.store_results()
# trainer.plot_training_results()

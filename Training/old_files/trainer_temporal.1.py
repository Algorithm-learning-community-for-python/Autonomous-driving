import os
from keras.utils import plot_model

from data_handler_temporal import DataHandler
from network_temporal import load_network
from visualizer import Visualizer
from data_configuration import Config

class Trainer:
    def __init__(self):
        self.conf = Config()

        self.atrX = ["Direction", "Image"]
        self.atrY = ["Steer"]

        self.network_handler = None
        self.history = None
        self.model = None
        self.visualizer = Visualizer()


        self.data_handler = None
        self.init_data_handler()
        # self.init_network()

    def init_data_handler(self):
        # Sets Data, dataX and dataY
        self.data_handler = DataHandler(atrX=self.atrX, atrY=self.atrY, train_valid_split=self.conf.train_valid_split)
        # Set dataX, dataY
        self.data_handler.set_XY_data(self.atrX, self.atrY, train_valid_split=False)
        # Set training_data, validation_data,
        self.data_handler.set_train_valid_split(self.conf.train_valid_split)
        # TrainX,TrainY, ValidX,ValidY
        self.data_handler.set_XY_data(self.atrX, self.atrY, train_valid_split=True)

    def init_network(self):
        self.network_handler = load_network(self.conf.input_size_data, self.conf.input_data)
        plot_model(self.network_handler.model, to_file="model.png")

    def train(self):
        self.model = self.network_handler.model
        
        train_dir = self.data_handler.trainX.Direction.values
        train_dir = self.data_handler.get_values_as_numpy_arrays(train_dir)

        valid_dir = self.data_handler.validX.Direction.values
        valid_dir = self.data_handler.get_values_as_numpy_arrays(valid_dir)

        train_img = self.data_handler.trainX.Image.values
        train_img = self.data_handler.get_values_as_numpy_arrays(train_img)

        valid_img = self.data_handler.validX.Image.values
        valid_img = self.data_handler.get_values_as_numpy_arrays(valid_img)

        train_y = self.data_handler.trainY.values
        valid_y = self.data_handler.validY.values
        
        self.model.compile(
            loss=self.conf.train_conf.loss,
            optimizer=self.conf.train_conf.optimizer,
            metrics=self.conf.train_conf.metrics
        )

        self.history = self.model.fit(
            [train_img, train_dir],
            train_y,
            validation_data=([valid_img, valid_dir], valid_y),
            epochs=self.conf.train_conf.epochs,
            batch_size=self.conf.train_conf.batch_size
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
trainer.data_handler.plot_data()
# trainer.train()
# trainer.store_results()
# trainer.plot_training_results()

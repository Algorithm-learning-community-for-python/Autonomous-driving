""" Module to train a neural network """
#pylint: disable=superfluous-parens

import argparse
from keras.optimizers import Adam
from Temporal.trainer import Trainer
from Temporal.Networks.network_temporal import load_network as load_network



def train_single():
    #lr = 0.0001
    #e = 10
    #b = 16
    #f = True
    print("##########################################################################")
    print("########################  NEW TRAINING   #################################")
    print("##########################################################################")
    """print("Training with: ")
    print("lr = " + str(lr))
    print("B = " + str(b))
    print("E = " + str(e))
    print("F = " + str(f))"""
    trainer = Trainer()
    #trainer.conf.train_conf.epochs = e
    #trainer.conf.train_conf.batch_size = b
    #trainer.conf.filter_input = f
    trainer.initialise_generator_and_net()
    trainer.train()
    trainer.save()

def train_multi():
    print("GRID SEARCH")
    recs = ["/Measurments/modified_recording.csv"] #"/Measurments/recording.csv", modified_recording
    batch_sizes = [16] #[8, 16, 32]
    epochs = [30]
    nets = [1]
    sigmoid = [False]#, True]
    filtering = [True]
    learning_rates = [0.001, 0.0005, 0.0001]  #[0.00001, 0.0001, 0.001]#, 0.00005, 0.0002]
    for recording_data in recs:
        for s in sigmoid:
            for f in filtering:
                for b in batch_sizes:
                    for e in epochs:
                        for lr in learning_rates:
                            for n in nets:
                                    
                                print("##########################################################################")
                                print("########################  NEW TRAINING   #################################")
                                print("##########################################################################")
                                print("Training with: ")
                                print("lr = " + str(lr))
                                print("B = " + str(b))
                                print("E = " + str(e))
                                print("Net = " + str(n))
                                trainer = Trainer()


                                trainer.conf.train_conf.epochs = e
                                trainer.conf.train_conf.batch_size = b
                                trainer.conf.filter_input = f
                                trainer.conf.recordings_path = recording_data
                                #trainer.conf.train_conf.lr = lr
                                #trainer.conf.train_conf.optimizer = Adam(lr=lr)
                                if s:
                                    trainer.conf.loss_functions = {
                                        "output_Throttle": "mse", #Might be better with binary_crossentropy
                                        "output_Brake": "binary_crossentropy",
                                        "output_Steer": "mse",
                                    }
                                    trainer.conf.activation_functions = {
                                        "output_Throttle": None, #Might be better with binary_crossentropy
                                        "output_Brake": "sigmoid",
                                        "output_Steer": None,
                                    }

                                trainer.initialise_generator_and_net()
                                trainer.train()
                                trainer.save()

def main():
    argparser = argparse.ArgumentParser(
        description='Training module for autonomous driving')
    argparser.add_argument(
        '-m', '--multi',
        metavar='M',
        default=0,
        type=int
    )
    args = argparser.parse_args()

    if args.multi == 1:
        train_multi()
    else:
        train_single()


if __name__ == '__main__':
    main()
""" Module to train a neural network """
#pylint: disable=superfluous-parens
#Todo: remove
import sys
sys.path.append("../../Training")
import warnings
warnings.filterwarnings("ignore")
import argparse
from keras.optimizers import Adam, SGD
from Spatiotemporal.trainer import Trainer
from Spatiotemporal.Networks.network import load_network as load_network
from Spatiotemporal.Networks.nvidia import load_network as load_nvidia
from Spatiotemporal.Networks.nvidiaa_v1 import load_network as load_nvidiaa_v1
from Spatiotemporal.Networks.nvidiaa_v2 import load_network as load_nvidiaa_v2
from Spatiotemporal.Networks.nvidiaa_v3 import load_network as load_nvidiaa_v3
from Spatiotemporal.Networks.nvidiaa_v4 import load_network as load_nvidiaa_v4
#from Spatiotemporal.Networks.nvidiaa_v5 import load_network as load_nvidiaa_v5





from Spatiotemporal.Networks.network_v2 import load_network as load_network_v2
from Spatiotemporal.Networks.xception import load_network as load_xception


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
    epochs = [20]
    nets = [1]
    sigmoid = [False]#, True]
    sequence_lengths = [4]
    filtering = [True]
    learning_rates = [0.00012]  #[0.00001, 0.0001, 0.001]#, 0.00005, 0.0002]
    #momentum = [0,0.5,1]
    #decay = [0,0.1, 0.2]
    #nesterov = [False, True, ]

    """
    #Test with noise
    trainer = Trainer()
    trainer.conf.add_noise = True
    trainer.initialise_generator_and_net()
    trainer.network_handler = load_nvidiaa_v1(trainer.conf)
    trainer.train()
    trainer.save()
    # Test sequence lengths
    for l in sequence_lengths:
        trainer = Trainer()
        trainer.conf.input_size_data["Sequence_length"] = l
        trainer.initialise_generator_and_net()
        trainer.network_handler = load_nvidiaa_v1(trainer.conf)
        trainer.train()
        trainer.save()"""

    #Test networks
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
                                trainer.conf.train_conf.lr = lr
                                trainer.conf.train_conf.optimizer = Adam(lr=lr)
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
                                if n == 0:
                                    trainer.network_handler = load_nvidia(trainer.conf)
                                if n == 1:
                                    trainer.network_handler = load_nvidiaa_v1(trainer.conf)
                                if n == 2:
                                    trainer.network_handler = load_nvidiaa_v2(trainer.conf)
                                if n == 3:
                                    trainer.network_handler = load_nvidiaa_v3(trainer.conf)
                                if n == 4:
                                    trainer.network_handler = load_nvidiaa_v4(trainer.conf)
                                #if n == 5:
                                #    trainer.network_handler = load_nvidiaa_v5(trainer.conf)

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

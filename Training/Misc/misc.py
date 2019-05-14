""" Misc functions used for training """
#pylint: disable=superfluous-parens

from keras.utils import plot_model
import cv2
import os
import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt

def get_model_memory_usage(batch_size, model):
    """ Calculates and returns an estimate of the model size """
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


def create_new_folder(path):
    last_folder = 0
    for folder in os.listdir(path):
        if int(folder) >= last_folder:
            last_folder = int(folder)+1
    folder = last_folder
    path = path + str(folder)

    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
        return None
    else:
        print("Successfully created the directory %s " % path)
        return folder



def save_results(trainer, path):
    """
    Saves training results and settings of a trained model to Stored_models and Current_models
    currently saving:
    - model.h5: The actual modell trained
    - model.png: The visualisation of the layers of the model
    - model.txt: Summary of the model
    - conf.txt: The configuration used for the training of the model
    - history.txt: The training history
    """
    
    folder = trainer.folder
    nh = trainer.network_handler
    history = trainer.history
    conf = trainer.conf

    cur_model_path = path + "Current_model/"
    path = path + "Stored_models/"
    if folder:
        path = path + str(folder)
    else:
        folder = create_new_folder(path)
        path = path + str(folder)

    print("Print storing results too the following path: ")
    print(path)
    # Store model
    nh.model.save(path + "/model.h5")
    nh.model.save(cur_model_path + "/model.h5")

    # Store image of model
    plot_model(nh.model, to_file=path + '/model.png')

    # Store model summary
    with open(path + "/model.txt", "w") as f:
        nh.model.summary(print_fn=lambda x: f.write(x + "\n"))
        f.write("Memory usage: " + \
                str(get_model_memory_usage(
                    conf.train_conf.batch_size,
                    nh.model)))
    f.close()

    # Store config
    f = open(path + "/conf.txt", "wb+")
    f.write(str(conf.train_conf.__dict__) + "\n")
    f.write(str(conf.input_data) + "\n")
    f.write(str(conf.output_data) + "\n")
    f.write(str(conf.input_size_data) + "\n")
    f.write("Top crop: + " + str(conf.top_crop ) + "\n")
    f.write("Bottom crop: + " + str(conf.bottom_crop ) + "\n")
    f.write("Filtering: + " + str(conf.filter_input ) + "\n")
    f.write("Filtering degree: + " + str(conf.filtering_degree ) + "\n")
    f.write("Loss_functions: " + str(conf.loss_functions) + "\n")
    f.write("Activation_functions: " + str(conf.activation_functions) + "\n")
    f.write("Recordings: " + str(conf.recordings_path) + "\n")
    f.close()

    # store plot of losses
    keys = []
    for key, values in history.history.iteritems():
        plt.plot(values)
        keys.append(key)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(keys, loc='upper right')
    plt.savefig(path + '/loss.png')
    plt.close()

    # Store training history as txt
    f = open(path + "/history.txt", "wb+")
    f.write(str(history.history) + "\n")
    f.close()


def get_image(path, frame):
    """ Returns image given a folder path and frame number """
    file_name = get_image_name(frame)
    img = cv2.imread(path + file_name)
    if len(img) == 0:
        print("Error fetching image")
    return img

def get_image_name(frame):
    """ Pads and adds png to the frame """
    frame_len = len(str(frame))
    pad = ''
    for _ in range(8 - frame_len):
        pad += '0'
    frame = str(frame)
    file_name = pad + frame + '.png'
    return file_name

def get_data_paths(data="Training_data"):
    """ Fetch all available folder paths and return them in sorted order """
    path = "../../" + data
    data_paths = []
    for folder in os.listdir(path):
        if folder == ".DS_Store" or folder == "store.h5":
            continue
        data_paths.append(path + "/" + folder)
    data_paths.sort(key=lambda a: int(a.split("/")[-1]))
    return data_paths 
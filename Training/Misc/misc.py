
from keras.utils import plot_model

import os
def get_model_memory_usage(batch_size, model):
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

def save_results(nh, history, conf, path, folder=None):
    cur_model_path = path + "Current_model/"
    path = path + "Stored_models/"
    if folder:
        folder = folder
    else:
        last_folder = 0
        for folder in os.listdir(path):
            if int(folder) >= last_folder:
                last_folder = int(folder)+1
        folder = last_folder

    path = path + str(folder)

    try:  
        os.mkdir(path)

    except OSError:  
        print ("Creation of the directory %s failed" % path)
    else:  
        print ("Successfully created the directory %s " % path)

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

    f = open(path + "/conf.txt", "wb+")

    # Store config
    f.write(str(conf.train_conf.__dict__) + "\n")

    # Store settings
    f.write(str(conf.input_data) + "\n")
    f.write(str(conf.input_size_data) + "\n")

    # Store training history
    f.write(str(history.history) + "\n")

    f.close()

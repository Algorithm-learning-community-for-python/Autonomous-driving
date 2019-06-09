from keras.models import Model
import matplotlib.pyplot as plt
from keras.models import load_model
from misc import get_data_paths, get_image
import cv2
import numpy as np
import pandas as pd
from Spatial.data_configuration import Config
from Spatial.batch_generator import BatchGenerator
import os
import keras.backend as K

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
tf_config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=tf_config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

conf = Config()
conf.train_conf.batch_size = 1
generator = BatchGenerator(conf, "Validation_data")
model = load_model("../../Training/Spatial/Current_model/" + os.listdir("../../Training/Spatial/Current_model/")[0])
temp = []
throttle_predictions = []
brake_predictions = []
steer_predictions = [] 
throttle_targets = []
brake_targets = []
steer_targets = [] 

predictions = []
targets = []
losses =  []
images = []
i = 0
for i in range(1000):#len(generator)):
    batch = generator[i]
    batchX = batch[0]
    batchY = batch[1]
    temp.append(batchX)
    images.append(batchX["input_Image"])
    pred = model.predict(batchX, batch_size=1)
    prediction =  {
        out.name.split(':')[0].encode("ascii").split("/")[0]: pred[i][0] for i, out in enumerate(model.outputs)
    }
    
    throttle = float(prediction["output_Throttle"][0])
    brake = float(prediction["output_Brake"][0])
    steering = float(prediction["output_Steer"][0])

    throttle_predictions.append(throttle)
    brake_predictions.append(brake)
    steer_predictions.append(steering)
    loss1 = K.mean(K.square(throttle - batchY["output_Throttle"]))
    loss2 = K.mean(K.square(steering - batchY["output_Steer"]))
    loss3 = K.mean(K.square(brake - batchY["output_Brake"]))

    loss = [loss1, loss2, loss3]
    losses.append(loss)
    predictions.append(pred)
    targets.append(batchY)


max_value = max(steer_predictions)
max_index = steer_predictions.index(max_value)

min_value = min(steer_predictions)
min_index = steer_predictions.index(min_value)

minum = 1
for i,prediction in enumerate(steer_predictions):
    cur = abs(prediction)
    if(minum > cur):
        minum = cur
        zero_index = i

def display_activation_cnn(activations, col_size, row_size, act_index, desc): 
    activation = activations[act_index]
    n_act = activation.shape[-1]
    matrix_size = int(np.ceil(np.sqrt(n_act)))
    if matrix_size < row_size:
        row_size = matrix_size
        col_size = matrix_size
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    row = 0
    col = 0
    fignr = 1
    for activation_index in range(activation.shape[-1]):
        ax[row][col].imshow(activation[0, :, :, activation_index])
        col += 1
        if col >= col_size:
            row += 1
            col = 0
            if row >= row_size and activation_index != activation.shape[-1]-1:
                row = 0
                fig.savefig('./images/activations/'+desc+ "_layer" +str(act_index)+"_fig-"+str(fignr)+'.png')
                fignr+=1

                fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    fig.savefig('./images/activations/'+desc+ "_layer" +str(act_index)+"_fig-"+str(fignr)+'.png')
    plt.close()
    #plt.show()
"""
def display_heatmap(x, loss, last_conv_layer):
    grads = K.gradients(loss, last_conv_layer)
    print(grads)
    pooled_grads = K.mean(grads)
    iterate = K.function([model.input], [pooled_grads, last_conv_layer[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    plt.show()
"""


X_train = temp
#print(model.summary())
layer_outputs = [layer.output for layer in model.layers][1:]
last = None
temp_layers = []
#print(layer_outputs)
for layer in layer_outputs:
    if "input" in str(layer):
        continue
    else:
        temp_layers.append(layer)
        if "conv" in str(layer):
            last = layer
            print(last)

activation_model = Model(inputs=model.input, outputs=temp_layers)


#display_heatmap(temp[max_index],  losses[max_index], last)

for act in range(6):
    image_index = max_index
    activations = activation_model.predict(X_train[image_index])
    print("IMAGE NR: " + str(image_index))
    print(predictions[image_index])
    #plt.imshow(X_train[image_index]+.5)
    cv2.imwrite('./images/activations/turn_right.png',(images[image_index]))    
    display_activation_cnn(activations, 6, 6, act, "steer_right")

for act in range(6):
    image_index = min_index
    activations = activation_model.predict(X_train[image_index])
    print("IMAGE NR: " + str(image_index))
    print(predictions[image_index])
    #plt.imshow(X_train[image_index]+.5)
    cv2.imwrite('./images/activations/turn_left.png',(images[image_index]))    

    display_activation_cnn(activations, 6, 6, act, "steer_left")

for act in range(6):
    image_index = zero_index
    activations = activation_model.predict(X_train[image_index])
    print("IMAGE NR: " + str(image_index))
    print(predictions[image_index])
    #plt.imshow(X_train[image_index]+.5)
    cv2.imwrite('./images/activations/go_straight.png',(images[image_index]))    

    display_activation_cnn(activations, 6, 6, act, "go_straight")

from keract import get_activations
from keract import display_activations
from keract import display_heatmaps

from keras.models import Model
from keras.models import load_model
from misc import get_data_paths, get_image
import numpy as np
import pandas as pd
from Spatial.data_configuration import Config
from Spatial.batch_generator import BatchGenerator
import os
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

i = 0
for i in range(len(generator)):
    i += 1
    if i > 10:
        break
        
    batch = generator[i]
    batchX = batch[0]
    batchY = batch[1]
    temp.append(batchX)
    prediction = model.predict(batchX, batch_size=1)
    prediction =  {
        out.name.split(':')[0].encode("ascii").split("/")[0]: prediction[i][0] for i, out in enumerate(model.outputs)
    }
    
    throttle = float(prediction["output_Throttle"][0])
    brake = float(prediction["output_Brake"][0])
    steering = float(prediction["output_Steer"][0])

    throttle_predictions.append(throttle)
    brake_predictions.append(brake)
    steer_predictions.append(steering)

    predictions.append(prediction)
    targets.append(batchY)

activations = get_activations(model, temp[0])
display_activations(activations, cmap="rgb", save=False)
#display_heatmaps(activations, input_image, save=False)

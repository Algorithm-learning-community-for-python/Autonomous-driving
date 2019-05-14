import timeit
import pandas as pd
from Misc.misc import get_image
import numpy as np
import random

def get_new_measurments_recording():
    """Loads measurments from the next recording"""
    #for i in range(2):
    i=0
    path = "../../Training_data/"+str(i)+"/Measurments/recording.csv"
    data = pd.read_csv(path)
    #Filter
    data = data.drop(
        data[(np.power(data.Steer, 2) < 0.001) & \
        random.randint(0, 10) > (10 - (10 * 1))].index)

    data = add_images(data, i)



def add_images(data, i):
    """Fetches image sequences for the current recording"""
    data["Image"] = None
    path ="../../Training_data/"+str(i)+"/Images/"
    for index, frame in enumerate(data.frame.values):
        i = 2
        img = get_image(path, frame)
        img = img[..., ::-1]
        img = img[100:, :, :]
        data.at[index, "Image"] = np.array(img)
    return data


print(timeit.timeit(get_new_measurments_recording, number=1))
#print(timeit.timeit(read_from_store, number=1))
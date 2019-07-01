#pylint: disable=superfluous-parens

from Temporal.batch_generator import BatchGenerator
import numpy as np
from Temporal.data_configuration import Config
conf = Config()
conf.train_conf.batch_size = 2
conf.filter_input = True
#generator = BatchGenerator(conf)
g = BatchGenerator(conf)
input_measures = [key for key in conf.available_columns if conf.input_data[key]]
output_measures = [key for key in conf.available_columns if conf.output_data[key]]

print("Length of generator: " + str(len(g)))
exit()

for b in range(len(g)):
    n = g[b]
    bx = n[0]
    by = n[1]
    #print(b)
    print(bx)
    print(by)
    """
    print("INPUT SAMPLE BATCH")
    for i in range(conf.input_size_data["Sequence_length"]):
        print("input_Image "+str(i) + "\n")
        print(bx["input_Image"+str(i)])
        print("\n")
    for measure in input_measures:
        for i in range(conf.input_size_data["Sequence_length"]):
            print("input_"+measure+str(i) + "\n")

            print(bx["input_"+measure+str(i)])
            print("\n")

    print("OUTPUT SAMPLE BATCH")

    for measure in output_measures:
        #if by["output_"+measure] is None:
        print("output_"+measure)
        print(measure + " " + str(by["output_"+measure]) + "\n")
    
    """
    raw_input()
"""
    sy = by[0]
    for sx in bx:
        sequence = sx[0]
        direction = sx[1]
        directions = sx[2]
        #if direction[2] != 1:
            #print(direction)
"""
"""

from data_configuration import Config
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
def get_one_hot_encoded( categories, values):
    ### FIT to categories
    label_encoder = LabelEncoder()
    label_encoder.fit(categories)
    integer_encoded = label_encoder.transform(categories)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoder = onehot_encoder.fit(integer_encoded)
    ### Encode values
    integer_encoded = label_encoder.transform(values)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.transform(integer_encoded)
    return onehot_encoded

    
c = Config()
ohe = get_one_hot_encoded(c.direction_categories, [["RoadOption.LANEFOLLOW"], ["RoadOption.LEFT"]])
print(ohe)
ohe = get_one_hot_encoded(c.direction_categories, ["RoadOption.LEFT"])
print(ohe)
ohe = get_one_hot_encoded(c.direction_categories, ["RoadOption.RIGHT"])
print(ohe)
ohe = get_one_hot_encoded(c.direction_categories, ["RoadOption.CHANGELANELEFT"])
print(ohe)
ohe = get_one_hot_encoded(c.direction_categories, ["RoadOption.CHANGELANERIGHT"])
print(ohe)

"""

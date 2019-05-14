#pylint: disable=superfluous-parens

from batch_generator import BatchGenerator
import numpy as np
from Spatial.data_configuration import Config
conf = Config()
conf.train_conf.batch_size = 1
generator = BatchGenerator(conf)
g = generator.generate()
input_measures = [key for key in conf.available_columns if conf.input_data[key]]
output_measures = [key for key in conf.available_columns if conf.output_data[key]]


for n in g:
    print("Progress:")
    print(generator.current_idx)
    print(generator.folder_index)
    bx = n[0]
    by = n[1]
    print(bx)
    """
    print("INPUT SAMPLE BATCH")
    print(bx)
    for measure in input_measures:
        print("inpu_"+measure)
        print(measure + " " + str(bx["input_"+measure]))

    print("OUTPUT SAMPLE BATCH")

    for measure in output_measures:
        print("output_"+measure)
        print(measure + " " + str(by["output_"+measure]))
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

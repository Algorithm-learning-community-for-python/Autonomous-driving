
from batch_generator import KerasBatchGenerator

"""
generator = KerasBatchGenerator(4)
g = generator.generate()
for n in g:
    #n = g.next()
    print("Progress:")
    print(generator.current_idx)
    print(generator.folder_index)
    bx = n[0]
    by = n[1]

    sx = bx[0]
    sy = by[0]
    sequence = sx[0]
    direction = sx[1]
    directions = sx[2]
    if direction[2] == 1:
        print(sequence.shape)
        print(directions)
    #print(sy)
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
"""NETWORK HANDLER AND CREATER"""
from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda,Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, concatenate

class NetworkHandler():
    """ Class that holds the network in use"""
    def __init__(self):
        self.conv_layers = 0
        self.pool_layers = 0
        self.batch_norm_layers = 0
        self.activation_layers = 0
        self.dropout_layers = 0
        self.dense_layers = 0
        self.model = None

    def conv_block(self, x, filters, kernel_size, stride, padding='SAME'):
        #print("CREATE CONV BLOCK WITH INPUT:")
        #print(x)
        x = self.conv(x, filters, kernel_size, stride, padding)
        #print("ADDED conv")
        #print(x)

        x = self.batch_norm(x)
        #print("ADDED batch_norm")
        #print(x)
        x = self.dropout(x, rate=0.0)
        #print("ADDED dropout")
        #print(x)
        x = self.activation(x)
        return x


    def conv(self, x, filters, kernel_size, stride, padding='same'):
        self.conv_layers += 1
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding=padding,  name="conv_"+str(self.conv_layers))(x)
        return x

    def max_pool(self, x, pool_size=(2,2)):
        self.pool_layers += 1
        x = MaxPooling2D(pool_size=pool_size, name="max_pool_" + str(self.conv_layers))(x)
        return x

    def batch_norm(self, x, chanDim=-1):
        #after a Conv2D layer with data_format="channels_first", set axis=1 in BatchNormalization
        self.batch_norm_layers += 1
        x = BatchNormalization(axis=chanDim, name="batch_norm_" + str(self.batch_norm_layers))(x)
        return x

    def activation(self, x, function="relu"):
        self.activation_layers += 1
        x = Activation(function, name="activation_" + str(self.activation_layers))(x)
        return x
    def dropout(self, x, rate=0.0):
        self.dropout_layers += 1
        x = Dropout(rate, name="dropout_" + str(self.dropout_layers))(x)
        return x

    def dense(self, x, output_size, function=None):
        self.dense_layers += 1
        x = Dense(output_size, activation=function)(x)
        return x

def hsv_convert(x):
    import tensorflow as tf
    return tf.image.rgb_to_hsv(x)  

def load_network(conf):
    """ Creates and returns a model using the class network handler """
    input_size_data, input_data = conf.input_size_data, conf.input_data
    inputs = []
    x = Input(shape=input_size_data["Image"], name="input_1")
    inputs.append(x)
    net = NetworkHandler()

    # RGB TO HSV
    x = Lambda(hsv_convert)(x)

    # CONV 1
    #x = net.conv_block(x, filters=32, kernel_size=5, stride=2, padding='VALID')
    x = net.conv_block(x, 32, 5, 2, padding='VALID')
    print(x)

    # CONV 2
    x = net.conv_block(x, 32, 3, 1, padding='VALID')
    print(x)

    # CONV 3
    x = net.conv_block(x, 64, 3, 2, padding='VALID')
    print(x)

    # CONV 4
    x = net.conv_block(x, 64, 3, 1, padding='VALID')
    print(x)

    # CONV 5
    x = net.conv_block(x, 128, 3, 2, padding='VALID')
    print(x)

    # CONV 6
    x = net.conv_block(x, 128, 3, 1, padding='VALID')
    print(x)

    # CONV 7
    x = net.conv_block(x, 256, 3, 1, padding='VALID')
    print(x)

    # CONV 8
    x = net.conv_block(x, 256, 3, 1, padding='VALID')
    print(x)

    # FLATTEN
    x = Flatten()(x)
    print(x)

    # Fully connected 1
    x = net.dense(x, 512)
    x = net.dropout(x, 0.3)
    print(x)

    # Fully connected 2
    x = net.dense(x, 512)
    x = net.dropout(x, 0.3)

    #######     INPUT DATA     #######


    # DIRECTION
    if input_data["Direction"]:
        direction = Input(input_size_data["Direction"], name="input_2")
        inputs.append(direction)
        # Fully connected 1
        direction = net.dense(direction, 32)
        direction = net.dropout(direction, 0.5)
        # Fully connected 2
        direction = net.dense(direction, 32)
        direction = net.dropout(direction, 0.5)
        #Concatinate
        x = concatenate([x, direction], 1)

    # SPEED
    if input_data["Speed"]:
        speed = Input(input_size_data["Speed"], name="input_3")
        inputs.append(speed)

        # Fully connected 1
        speed = net.dense(speed, 128)
        speed = net.dropout(speed, 0.5)
        # Fully connected 2
        speed = net.dense(speed, 128)
        speed = net.dropout(speed, 0.5)
        #Concatinate
        x = concatenate([x, speed], 1)

    # Traffic Light
    if input_data["TL_state"]:
        tl = Input(input_size_data["TL_state"], name="input_traffic_light")
        inputs.append(tl)
        # Fully connected 1
        tl = net.dense(tl, 128)
        tl = net.dropout(tl, 0.5)
        # Fully connected 2
        tl = net.dense(tl, 128)
        tl = net.dropout(tl, 0.5)
        #Concatinate
        x = concatenate([x, tl], 1)

    # Speed limit
    if input_data["speed_limit"]:
        speed_limit = Input(input_size_data["speed_limit"], name="input_speed_limit")
        inputs.append(speed_limit)
        # Fully connected 1
        speed_limit = net.dense(speed_limit, 128)
        speed_limit = net.dropout(speed_limit, 0.5)
        # Fully connected 2
        speed_limit = net.dense(speed_limit, 128)
        speed_limit = net.dropout(speed_limit, 0.5)
        #Concatinate
        x = concatenate([x, speed_limit], 1)


    x = net.dense(x, 512)
    x = net.dropout(x, 0.5)

    x = net.dense(x, 128)
    x = net.dropout(x, 0.5)

    x = net.dense(x, 32)
    x = Dense(input_size_data["Output"], name="output")(x)

    net.model = Model(inputs=inputs, outputs=x)

    return net
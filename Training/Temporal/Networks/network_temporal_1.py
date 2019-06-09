from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda,Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, concatenate, LSTM
from keras.layers import TimeDistributed

class NetworkHandler():
    def __init__(self):
        self.conv_layers = 0
        self.pool_layers = 0
        self.batch_norm_layers = 0
        self.activation_layers = 0
        self.dropout_layers = 0
        self.dense_layers = 0
        self.lstm_layers = 0
        self.model = None

    def conv_block(self, x, filters, kernel_size, stride, padding='SAME'):
        x = self.conv(x, filters, kernel_size, stride, padding)
        x = self.batch_norm(x)
        x = self.dropout(x, rate=0.0)
        x = self.activation(x)
        return x

    def conv(self, x, filters, kernel_size, stride, padding='valid', activation=None):
        self.conv_layers += 1
        x = TimeDistributed(Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            activation=activation,
            name="conv_" + str(self.conv_layers)))(x)
        return x

    def max_pool(self, x, pool_size=(2, 2)):
        self.pool_layers += 1
        x = TimeDistributed(MaxPooling2D(pool_size=pool_size, name="max_pool_" + str(self.conv_layers)))(x)
        return x

    def batch_norm(self, x, chanDim=-1):
        # after a Conv2D layer with data_format="channels_first", set axis=1 in BatchNormalization
        self.batch_norm_layers += 1
        x = TimeDistributed(BatchNormalization(axis=chanDim, name="batch_norm_" + str(self.batch_norm_layers)))(x)
        return x

    def activation(self, x, activation_function="relu"):
        self.activation_layers += 1
        x = TimeDistributed(Activation(activation_function, name="activation_" + str(self.activation_layers)))(x)
        return x

    def dropout(self, x, rate=0.0, name=None, td=True):
        self.dropout_layers += 1
        if not td:
            x = Dropout(rate, name="dropout_" + str(self.dropout_layers))(x)
        else:
            x = TimeDistributed(Dropout(rate, name="dropout_" + str(self.dropout_layers)), name=name)(x)
        return x

    def dense(self, x, output_size, td=True, activation_function=None, name=None):
        """ Adds a dense layer """
        if not name:
            name="dense_" + str(self.dense_layers)
        self.dense_layers += 1
        if not td:
            x = Dense(output_size, activation=activation_function, name=name)(x)
        else:
            x = TimeDistributed(Dense(output_size, activation=activation_function), name=name)(x)
        return x

    def lstm(self, x, output_size, return_sequences=False):
        self.lstm_layers += 1
        x = LSTM(output_size, return_sequences=return_sequences, dropout=0.5)(x)
        return x


def hsv_convert(x):
    import tensorflow as tf
    return tf.image.rgb_to_hsv(x)  


def load_network(conf):
    input_measures = [key for key in conf.available_columns if conf.input_data[key]]
    output_measures = [key for key in conf.available_columns if conf.output_data[key]]
    input_size_data = conf.input_size_data
    inputs = []
    x = Input(shape=[input_size_data["Sequence_length"]] + input_size_data["Image"], name="input_Image")
    inputs.append(x)
    net = NetworkHandler()

    # RGB TO HSV
    x = TimeDistributed(Lambda(hsv_convert))(x)

    # CONV 1
    x = net.conv(x, 24, 5, 2, activation="relu")
        #x = net.conv(x, 24, 5, 2, activation="relu")

    # CONV 2
    x = net.conv(x, 36, 5, 2, activation="relu")

    # CONV 3
    x = net.conv(x, 48, 5, 2, activation="relu")

    # CONV 4
    x = net.conv(x, 64, 3, 1, activation="relu")

    # CONV 4
    x = net.conv(x, 64, 3, 1, activation="relu")

    # FLATTEN
    x = TimeDistributed(Flatten(), name="time_flatten")(x)

    # Fully connected 1
    #x = net.dense(x, 512, name="time_FC1")
    #x = net.dropout(x, 0.3)
    #print(x)
    x = net.lstm(x, 500, return_sequences=False)

    # Fully connected 2
    #x = net.dense(x, 512, name="time_FC2")
    #x = net.dropout(x, 0.3)

    # ######     INPUT DATA     ###### #
    #######     INPUT DATA     #######
    #print(x)
    for measure in input_measures:
        #if measure == "Speed":
         #   input_layer = Input([input_size_data["Sequence_length"]], name="input_" + measure)
         #   inputs.append(input_layer)
         #   x = concatenate([x, input_layer], 1)
        #else:
        input_layer = Input(input_size_data[measure], name="input_" + measure)
        inputs.append(input_layer)
       # input_layer = net.dense(input_layer, 10, activation_function="relu")
        x = concatenate([x, input_layer])
    #print(x)
 
    #x = net.lstm(x, 100, return_sequences=True)
    #x = net.dropout(x, 0.5)
    #x = net.lstm(x, 50, return_sequences=True)
    #x = net.lstm(x, 10, return_sequences=False)

    x = net.dense(x, 100, td = False, activation_function="relu")
    x = net.dense(x, 50, td = False, activation_function="relu")
    x = net.dense(x, 10, td = False)

    outputs = []
    for measure in output_measures:
        output_layer = net.dense(
            x,
            conf.output_size_data[measure],
            td=False,
            activation_function=conf.activation_functions["output_" + measure],
            name="output_" + measure
            )
        outputs.append(output_layer)    #x = net.dense(x, input_size_data["Output"], td = False)

    net.model = Model(inputs=inputs, outputs=outputs)
    #print(net.model.summary())

    return net

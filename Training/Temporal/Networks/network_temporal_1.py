""" module used to create and hold a network model """
#pylint: disable=invalid-name
from keras.models import Model
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, concatenate, LSTM, CuDNNLSTM
from keras.layers import TimeDistributed

class NetworkHandler(object):
    """ Class that holds the network in use"""
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
        """ Creates a full conv block """
        x = self.conv(x, filters, kernel_size, stride, padding)
        x = self.batch_norm(x)
        x = self.dropout(x, rate=0.0)
        x = self.activation(x)
        return x

    def conv(self, x, filters, kernel_size, stride, padding='valid', activation=None):
        """ Adds a convolutional layer """
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
        """ Adds a max pooling layer """
        self.pool_layers += 1
        x = TimeDistributed(
            MaxPooling2D(pool_size=pool_size, name="max_pool_" + str(self.conv_layers))
        )(x)
        return x

    def batch_norm(self, x, chanDim=-1):
        """
        Adds a batch normalisation layer
        NB: after a Conv2D layer with data_format="channels_first",
        set axis=1 in BatchNormalization
        """
        self.batch_norm_layers += 1
        x = TimeDistributed(
            BatchNormalization(axis=chanDim, name="batch_norm_" + str(self.batch_norm_layers))
        )(x)
        return x

    def activation(self, x, activation_function="relu"):
        """ Adds an activation-layer """
        self.activation_layers += 1
        x = TimeDistributed(
            Activation(activation_function, name="activation_" + str(self.activation_layers))
        )(x)
        return x

    def dropout(self, x, rate=0.0, name=None, td=True):
        """ Adds a dropout layer """
        self.dropout_layers += 1
        if not td:
            x = Dropout(rate, name="dropout_" + str(self.dropout_layers))(x)
        else:
            x = TimeDistributed(
                Dropout(rate, name="dropout_" + str(self.dropout_layers)),
                name=name
            )(x)
        return x

    def dense(self, x, output_size, td=True, activation_function=None, name=None):
        """ Adds a dense layer """
        if not name:
            name = "dense_" + str(self.dense_layers)
        self.dense_layers += 1
        if not td:
            x = Dense(output_size, activation=activation_function, name=name)(x)
        else:
            x = TimeDistributed(Dense(output_size, activation=activation_function), name=name)(x)
        return x

    def lstm(self, x, output_size, return_sequences=False, activation="tanh", dropout=0):
        """ Adds a lstm layer """
        self.lstm_layers += 1
        x = LSTM(
            output_size,
            activation=activation,
            return_sequences=return_sequences,
            dropout=dropout
        )(x)
        return x

    def cudnnlstm(self, x, output_size, return_sequences=False):
        """ Adds a CuDNNLSTM layer """
        self.lstm_layers += 1
        x = CuDNNLSTM(output_size, return_sequences=return_sequences)(x)
        return x


def hsv_convert(x):
    """ Converts input from rgb to hsv in the range 0-1 """
    import tensorflow as tf
    return tf.image.rgb_to_hsv(x)


def load_network(conf):
    """ Creates and returns a model using the class network handler """
    input_measures = [key for key in conf.available_columns if conf.input_data[key]]
    output_measures = [key for key in conf.available_columns if conf.output_data[key]]
    input_size_data = conf.input_size_data
    seq_len = input_size_data["Sequence_length"]
    inputs = []
    x = Input(shape=[seq_len] + input_size_data["Image"], name="input_Image")
    inputs.append(x)
    net = NetworkHandler()

    # RGB TO HSV
    x = TimeDistributed(Lambda(hsv_convert))(x)

    # CONV 1
    x = net.conv(x, 24, 5, 2, activation="tanh")
    #x = net.batch_norm(x)
    # CONV 2
    x = net.conv(x, 36, 5, 2, activation="tanh")
    #x = net.batch_norm(x)

    # CONV 3
    x = net.conv(x, 48, 5, 2, activation="tanh")
    #x = net.batch_norm(x)
    #x = net.dropout(x, 0.5)

    # CONV 4
    x = net.conv(x, 64, 3, 1, activation="tanh")
    #x = net.batch_norm(x)

    # CONV 5
    x = net.conv(x, 64, 3, 1, activation="tanh")

    # FLATTEN
    x = TimeDistributed(Flatten(), name="time_flatten")(x)
    #x = net.batch_norm(x)


    #######     INPUT DATA     #######
    for measure in input_measures:
        input_layer = Input([seq_len] + input_size_data[measure], name="input_" + measure)
        inputs.append(input_layer)
        x = concatenate([x, input_layer])

    #x = net.dense(x, 100, activation_function="relu")
    x = net.lstm(x, 100, return_sequences=False)
    x = net.dense(x, 50, td=False, activation_function="tanh")
    x = net.dense(x, 32, td=False)

    outputs = []
    for measure in output_measures:
        output_layer = net.dense(
            x,
            conf.output_size_data[measure],
            td=False,
            activation_function=conf.activation_functions["output_" + measure],
            name="output_" + measure
            )
        outputs.append(output_layer)

    net.model = Model(inputs=inputs, outputs=outputs)

    return net

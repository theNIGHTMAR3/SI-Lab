from dataclasses import dataclass
from typing import List
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.keras import layers


def build_default_regression_model(num_inputs, num_outputs):
    # Przykładowa prosta sieć neuronowa do zadania regresji

    model = tf.keras.Sequential()
    model.add(layers.Dense(20, activation='relu', input_shape=[num_inputs], name='ukryta_1'))
    model.add(layers.Dense(40, activation='relu', name='ukryta_2'))
    model.add(layers.Dense(30, activation='relu', name='ukryta_3'))
    model.add(layers.Dense(num_outputs, name='wyjsciowa'))  # model regresyjny, activation=None w warstwie wyjściowej

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.mse,
                  metrics=[tf.keras.metrics.mean_absolute_error, 'mse'])

    return model


@dataclass
class FFN_Hyperparams:
    # stale (zalezne od problemu)
    num_inputs: int
    num_outputs: int

    # do znalezienia optymalnych (np. metodą Random Search) [w komentarzu zakresy wartości, w których można szukać]
    hidden_dims: List[int]  # [10] -- [100, 100, 100]
    activation_fcn: str  # wybor z listy, np: ['relu', 'tanh', 'sigmoid', 'elu', 'selu', 'softplus']
    learning_rate: float  # od 0.1 do 0.00001 (losowanie eksponensu 10**x, gdzie x in [-5, -1])


def build_model(hp: FFN_Hyperparams):
    dims = hp.hidden_dims
    activation = hp.activation_fcn
    learning = hp.learning_rate
    inputs = hp.num_inputs
    outputs = hp.num_outputs

    model = tf.keras.Sequential()

    model.add(layers.Dense(dims[0], activation=activation, input_shape=[inputs], name='ukryta_1'))

    if len(dims) > 1:
        for i in range(1, len(dims)):
            temp_name = "ukryta" + str(i)
            model.add(layers.Dense(dims[i], activation=activation, name=temp_name))

    model.add(layers.Dense(outputs, name='wyjsciowa'))  # model regresyjny, activation=None w warstwie wyjściowej

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.mse,
                  metrics=[tf.keras.metrics.mean_absolute_error, 'mse'])

    return model

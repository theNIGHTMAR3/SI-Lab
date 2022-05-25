import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import json
import dataclasses
import os
import random as r
from part1_data_preparation import load_data, standardization
from part2_model_definition import build_model, FFN_Hyperparams


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()


def train(model, train_data, valid_data=None, exp_dir='exp_00'):

    (x, y) = train_data

    # TODO define callbacks EarlyStopping, ModelCheckpoint, TensorBoard
    # my callbacks
    es_cbk = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=20)

    ckp_cbk = tf.keras.callbacks.ModelCheckpoint(os.path.join(exp_dir, 'model_best_weights'),
                                                 monitor='val_loss',
                                                 verbose=0,
                                                 save_best_only=True,
                                                 save_weights_only=True,
                                                 mode='min')
    tb_cbk = tf.keras.callbacks.TensorBoard(os.path.join(exp_dir,"tensorboard.log"))

    #es_cbk,ckp_cbk,tb_cbk

    # training
    if valid_data is None:
        history = model.fit(x=x, y=y, batch_size=64, validation_split=0.2,
                            epochs=1000, verbose=1, callbacks=[es_cbk,ckp_cbk,tb_cbk])
    else:
        history = model.fit(x=x, y=y, batch_size=64, validation_data=valid_data,
                            epochs=1000, verbose=1, callbacks=[es_cbk,ckp_cbk,tb_cbk])

    return model, history


def single_run(hp, train_data, experiment_dir, valid_data=None, verbose=True):

    if verbose:
        print(f"Training model with hyperparams: {hp}")

    os.makedirs(experiment_dir, exist_ok=True)

    #save hyperparams
    with open(os.path.join(experiment_dir, 'hp.json'), 'w') as f:
        json.dump(dataclasses.asdict(hp), f)

    # build model
    model = build_model(hp)
    model.build((None, hp.num_inputs))
    if verbose:
        print(model.summary())

    # train model
    return train(model, train_data, valid_data=valid_data, exp_dir=experiment_dir)


def data_split(features, targets, valid_ratio=0.2):
    valid_mask = np.random.choice([False, True], features.shape[0], p=[1 - valid_ratio, valid_ratio])
    return (features[~valid_mask, :], targets[~valid_mask]), (features[valid_mask, :], targets[valid_mask])


def get_random_hp(constants):
    hidden_dims_len_range = [1, 3]
    hidden_dims_values_range = [10, 100]
    activation_fcn_choices = ['relu', 'tanh', 'sigmoid', 'elu', 'selu', 'softplus']
    lr_exponent_range = [-5, -1]

    # TODO randomly select hyperparameters from given ranges
    #raise NotImplementedError

    random_dims=[]

    for i in range(r.choice(hidden_dims_len_range)):
        random_dims.append(r.choice(hidden_dims_values_range))

    return FFN_Hyperparams(constants[0], constants[1], random_dims, r.choice(activation_fcn_choices), r.choice(lr_exponent_range))


if __name__ == '__main__':

    train_features, train_targets = load_data('data_new/train_data.npz')
    train_features, standardizer = standardization(train_features)

    # save standardizer
    with open('data_new/standardizer.pkl', 'wb') as f:
        pickle.dump(standardizer, f, pickle.HIGHEST_PROTOCOL)

    random_search = False

    if not random_search:
        experiment_dir = 'exp_single'

        # setup hyperparameters
        hp = FFN_Hyperparams(train_features.shape[1], 1, [10, 20, 10], 'relu', 0.001)

        # run training with them
        _, history = single_run(hp, (train_features, train_targets), experiment_dir)
        plot_loss(history)
    else:
        # split data into train and valid
        (train_features, train_targets), (valid_features, valid_targets) = data_split(train_features, train_targets)

        best_val_loss = np.inf
        best_experiment = None

        for n in range(5):
            print(f"Uruchomienie treningu nr {n}/10")

            experiment_dir = f'exp_{n:02d}'

            # setup hyperparameters
            hp = get_random_hp(constants=(train_features.shape[1], 1))

            # run training with them
            _, history = single_run(hp, (train_features, train_targets), experiment_dir,
                                    valid_data=(valid_features, valid_targets))

            min_val_loss = min(history.history['val_loss'])
            if min_val_loss < best_val_loss:
                best_val_loss = min_val_loss
                best_experiment = experiment_dir

        print(f"Koniec procedury Random Search.")
        print(f"Najlepiej wypadł eksperyment {best_experiment} (val_loss={best_val_loss:0.4f}).")




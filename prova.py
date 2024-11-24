import pandas as pd
import numpy as np
from scipy import ndimage, fft
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler

#import keras

#from keras.models import Sequential

#from keras.layers import Dense, Activation, Dropout

#from keras.layers.normalization import BatchNormalization

#from keras import metrics

#from keras.callbacks import ModelCheckpoint

from imblearn.over_sampling import SMOTE

from pathlib import Path

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import math
import time

from sklearn.metrics import classification_report

from scipy import ndimage, fft
from sklearn.preprocessing import normalize

import tensorflow as tf



class LightFluxProcessor:

    def __init__(self, fourier=True, normalize=True, gaussian=True, standardize=True):
        self.fourier = fourier
        self.normalize = normalize
        self.gaussian = gaussian
        self.standardize = standardize

    def fourier_transform(self, X):
        return np.abs(fft.fft(X, n=X.size))

    def process(self, df_train_x, df_dev_x):
        # Apply fourier transform
        if self.fourier:
            print("Applying Fourier...")
            shape_train = df_train_x.shape
            shape_dev = df_dev_x.shape
            df_train_x = df_train_x.apply(self.fourier_transform, axis=1)
            df_dev_x = df_dev_x.apply(self.fourier_transform, axis=1)

            df_train_x_build = np.zeros(shape_train)
            df_dev_x_build = np.zeros(shape_dev)

            for ii, x in enumerate(df_train_x):
                df_train_x_build[ii] = x

            for ii, x in enumerate(df_dev_x):
                df_dev_x_build[ii] = x

            df_train_x = pd.DataFrame(df_train_x_build)
            df_dev_x = pd.DataFrame(df_dev_x_build)

            # Keep first half of data as it is symmetrical after previous steps
            df_train_x = df_train_x.iloc[:, : (df_train_x.shape[1] // 2)].values
            df_dev_x = df_dev_x.iloc[:, : (df_dev_x.shape[1] // 2)].values

        # Normalize
        if self.normalize:
            print("Normalizing...")
            df_train_x = pd.DataFrame(normalize(df_train_x))
            df_dev_x = pd.DataFrame(normalize(df_dev_x))

            # df_train_x = df_train_x.div(df_train_x.sum(axis=1), axis=0)
            # df_dev_x = df_dev_x.div(df_dev_x.sum(axis=1), axis=0)

        # Gaussian filter to smooth out data
        if self.gaussian:
            print("Applying Gaussian Filter...")
            df_train_x = ndimage.filters.gaussian_filter(df_train_x, sigma=10)
            df_dev_x = ndimage.filters.gaussian_filter(df_dev_x, sigma=10)

        if self.standardize:
            # Standardize X data
            print("Standardizing...")
            std_scaler = StandardScaler()
            df_train_x = std_scaler.fit_transform(df_train_x)
            df_dev_x = std_scaler.transform(df_dev_x)

        print("Finished Processing!")
        return df_train_x, df_dev_x

np.random.seed(1)

LOAD_MODEL = True  # continue training previous weights or start fresh
RENDER_PLOT = False  # render loss and accuracy plots


def build_network(shape):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    return model


def np_X_Y_from_df(df):
    df = shuffle(df)
    df_X = df.drop(["LABEL"], axis=1)
    X = np.array(df_X)
    Y_raw = np.array(df["LABEL"]).reshape((len(df["LABEL"]), 1))
    Y = Y_raw == 2
    return X, Y


if __name__ == "__main__":
    train_dataset_path = r"metterepath\data_no_injected/exoTrain.csv"
    dev_dataset_path = r"metterepath\data_no_injected/exoTest.csv"

    print("Loading datasets...")
    df_train = pd.read_csv(train_dataset_path, encoding="ISO-8859-1")
    df_dev = pd.read_csv(dev_dataset_path, encoding="ISO-8859-1")
    
    # Generate X and Y dataframe sets
    df_train_x = df_train.drop("LABEL", axis=1)
    df_dev_x = df_dev.drop("LABEL", axis=1)
    df_train_y = df_train.LABEL
    df_dev_y = df_dev.LABEL

    # Process dataset
    LFP = LightFluxProcessor(
        fourier=True, normalize=True, gaussian=True, standardize=True
    )
    df_train_x, df_dev_x = LFP.process(df_train_x, df_dev_x)

    # Rejoin X and Y
    df_train_processed = pd.DataFrame(df_train_x).join(pd.DataFrame(df_train_y))
    df_dev_processed = pd.DataFrame(df_dev_x).join(pd.DataFrame(df_dev_y))

    # Load X and Y numpy arrays
    X_train, Y_train = np_X_Y_from_df(df_train_processed)
    X_dev, Y_dev = np_X_Y_from_df(df_dev_processed)
    
    
    # Print data set stats
    (num_examples, n_x) = (
        X_train.shape
    )  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[1]  # n_y : output size
    print("X_train.shape: ", X_train.shape)
    print("Y_train.shape: ", Y_train.shape)
    print("X_dev.shape: ", X_dev.shape)
    print("Y_dev.shape: ", Y_dev.shape)
    print("n_x: ", n_x)
    print("num_examples: ", num_examples)
    print("n_y: ", n_y)

    # Build model
    model = build_network(X_train.shape[1:])

    # Load weights
    load_path = ""
    my_file = Path(load_path)
    if LOAD_MODEL and my_file.is_file():
        model.load_weights(load_path)
        print("------------")
        print("Loaded saved weights")
        print("------------")

    sm = SMOTE()
    X_train_sm, Y_train_sm = sm.fit_resample(X_train, Y_train)
    # X_train_sm, Y_train_sm = X_train, Y_train
    
    # Train
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]
    print("Training...")
    history = model.fit(X_train_sm, Y_train_sm, epochs=50, batch_size=32)


    # Metrics
    train_outputs = model.predict(X_train, batch_size=32)
    dev_outputs = model.predict(X_dev, batch_size=32)
    train_outputs = np.rint(train_outputs)
    dev_outputs = np.rint(dev_outputs)
    accuracy_train = accuracy_score(Y_train, train_outputs)
    accuracy_dev = accuracy_score(Y_dev, dev_outputs)
    precision_train = precision_score(Y_train, train_outputs)
    precision_dev = precision_score(Y_dev, dev_outputs)
    recall_train = recall_score(Y_train, train_outputs)
    recall_dev = recall_score(Y_dev, dev_outputs)
    confusion_matrix_train = confusion_matrix(Y_train, train_outputs)
    confusion_matrix_dev = confusion_matrix(Y_dev, dev_outputs)


    # # Save model
    # print("Saving model...")
    # save_weights_path = "checkpoints_v2/weights-recall-{}-{}.weights.h5".format(
    #     recall_train, recall_dev
    # )  # load_path
    # model.save_weights(save_weights_path)
    # save_path = "models_v2/model-recall-{}-{}.weights.h5".format(
    #     recall_train, recall_dev
    # )  # load_path
    # # model.save(save_path)
    
    print("train set error", 1.0 - accuracy_train)
    print("dev set error", 1.0 - accuracy_dev)
    print("------------")
    print("precision_train", precision_train)
    print("precision_dev", precision_dev)
    print("------------")
    print("recall_train", recall_train)
    print("recall_dev", recall_dev)
    print("------------")
    print("confusion_matrix_train")
    print(confusion_matrix_train)
    print("confusion_matrix_dev")
    print(confusion_matrix_dev)
    print("------------")
    print("Train Set Positive Predictions", np.count_nonzero(train_outputs))
    print("Dev Set Positive Predictions", np.count_nonzero(dev_outputs))
    #  Predicting 0's will give you error:
    print("------------")
    print("All 0's error train set", 37 / 5087)
    print("All 0's error dev set", 5 / 570)

    print("------------")
    print("------------")
    
    
    if RENDER_PLOT:
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history["accuracy"])
        # plt.plot(history.history['val_acc'])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()

        # summarize history for loss
        plt.plot(history.history["loss"])
        # plt.plot(history.history['val_loss'])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()
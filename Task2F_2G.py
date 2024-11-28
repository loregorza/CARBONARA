import pandas as pd
import numpy as np
from scipy import ndimage, fft
from scipy.fftpack import fft
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import os

'''Change the local path before running the code'''

'''Task2F'''
#root_dir = r"C:\Users\loren\Documents\ComputationalAstrophysics\data_no_injection"

'''Task2G'''
root_dir = r"C:\Users\loren\Documents\ComputationalAstrophysics\data_injected"

np.random.seed(1)

LOAD_MODEL = True  # Continue training previous weights or start fresh
#RENDER_PLOT = True  # Render loss and accuracy plots


class LightFluxProcessor:
    def __init__(self, fourier=True, normalize=True, gaussian=True, standardize=True):
      
        self.fourier = fourier
        self.normalize = normalize
        self.gaussian = gaussian
        self.standardize = standardize

    def fourier_transform(self, X):
        
        X = np.asarray(X)  # Converts input to numpy array if necessary
        return np.abs(fft(X, n=len(X)))  # Use len(X) for compatibility with Series

    def process(self, df_train_x, df_dev_x):
        """
        Process training and development datasets using Fourier transform,
        normalization, Gaussian filtering, and standardization.
        """
        if self.fourier:
            print("Applying Fourier...")
            # Apply Fourier transform to each row
            df_train_x = df_train_x.apply(self.fourier_transform, axis=1)
            df_dev_x = df_dev_x.apply(self.fourier_transform, axis=1)

            # Rebuild DataFrames with transformed rows
            df_train_x = pd.DataFrame(np.vstack(df_train_x))
            df_dev_x = pd.DataFrame(np.vstack(df_dev_x))

            # Retain only the first half of the Fourier spectrum
            df_train_x = df_train_x.iloc[:, : df_train_x.shape[1] // 2]
            df_dev_x = df_dev_x.iloc[:, : df_dev_x.shape[1] // 2]

        if self.normalize:
            print("Normalizing...")
            df_train_x = normalize(df_train_x.values)
            df_dev_x = normalize(df_dev_x.values)

        if self.gaussian:
            print("Applying Gaussian Filter...")
            df_train_x = gaussian_filter(df_train_x, sigma=10)
            df_dev_x = gaussian_filter(df_dev_x, sigma=10)

        if self.standardize:
            print("Standardizing...")
            std_scaler = StandardScaler()
            df_train_x = std_scaler.fit_transform(df_train_x)
            df_dev_x = std_scaler.transform(df_dev_x)

        print("Finished Processing!")
        return pd.DataFrame(df_train_x), pd.DataFrame(df_dev_x)


def build_network(shape,n):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape),
            tf.keras.layers.Flatten(),
            #ciclo for forse, aumentanto il numero di neuroni aumenta la probabilità di overfitting
            tf.keras.layers.Dense(n, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),#questo è l output quindi rimane, il dropout deve essere prima del sigmoid 
        ]
    )

    #Aggiungere qui hidden layers con relu e dropout layer (tf.keras.layers.Dense(n_neuron, activation="relu"))
    # 
    
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

    train_dataset_path = os.path.join(root_dir, "./exoTrain.csv")
    dev_dataset_path = os.path.join(root_dir, "./exoTest.csv")

    print("Loading datasets...")
    df_train = pd.read_csv(train_dataset_path, encoding="ISO-8859-1")
    df_dev = pd.read_csv(dev_dataset_path, encoding="ISO-8859-1")

    # Generate X and Y dataframes
    df_train_x = df_train.drop("LABEL", axis=1)
    df_dev_x = df_dev.drop("LABEL", axis=1)
    df_train_y = df_train.LABEL
    df_dev_y = df_dev.LABEL

    # Process dataset
    LFP = LightFluxProcessor(fourier=True, normalize=True, gaussian=True, standardize=True)
    df_train_x, df_dev_x = LFP.process(df_train_x, df_dev_x)

    # Rejoin X and Y
    df_train_processed = pd.DataFrame(df_train_x).join(pd.DataFrame(df_train_y))
    df_dev_processed = pd.DataFrame(df_dev_x).join(pd.DataFrame(df_dev_y))

    # Load X and Y numpy arrays
    X_train, Y_train = np_X_Y_from_df(df_train_processed)
    X_dev, Y_dev = np_X_Y_from_df(df_dev_processed)

    # Print dataset stats
    print(f"X_train.shape: {X_train.shape}")
    print(f"Y_train.shape: {Y_train.shape}")
    print(f"X_dev.shape: {X_dev.shape}")
    print(f"Y_dev.shape: {Y_dev.shape}")


    g=[1,10,100]

    
    # Build model
    for i in range(len(g)):
        model = build_network(X_train.shape[1:],g[i])
        
    
        # Load weights if available
        load_path = ""
        if LOAD_MODEL and Path(load_path).is_file():
            model.load_weights(load_path)
            print("Loaded saved weights")
    
        # Apply SMOTE for balancing
        sm = SMOTE()
        X_train_sm, Y_train_sm = sm.fit_resample(X_train, Y_train)
    
        # Train the model
        print("Training...")
        history = model.fit(X_train_sm, Y_train_sm, epochs=50, batch_size=32)
    
        # Evaluate the model
        train_outputs = np.rint(model.predict(X_train, batch_size=32))
        dev_outputs = np.rint(model.predict(X_dev, batch_size=32))
        accuracy_train = accuracy_score(Y_train, train_outputs)
        accuracy_dev = accuracy_score(Y_dev, dev_outputs)
        precision_train = precision_score(Y_train, train_outputs)
        precision_dev = precision_score(Y_dev, dev_outputs)
        recall_train = recall_score(Y_train, train_outputs)
        recall_dev = recall_score(Y_dev, dev_outputs)
        confusion_matrix_train = confusion_matrix(Y_train, train_outputs)
        confusion_matrix_dev = confusion_matrix(Y_dev, dev_outputs)
        print('numero di neuroni', g[i])
        print(f"Train set error: {1.0 - accuracy_train}")
        print(f"Dev set error: {1.0 - accuracy_dev}")
        print("------------")
        print(f"Precision Train: {precision_train}")
        print(f"Precision Dev: {precision_dev}")
        print("------------")
        print(f"Recall Train: {recall_train}")
        print(f"Recall Dev: {recall_dev}")
        print("------------")
        print("Confusion Matrix Train:")
        print(confusion_matrix_train)
        print("Confusion Matrix Dev:")
        print(confusion_matrix_dev)
    
        # Plot accuracy
        plt.plot(history.history["accuracy"])
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train"], loc="right")
        plt.show()
        # Plot loss
        plt.plot(history.history["loss"])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train"], loc="upper right")
        plt.show()

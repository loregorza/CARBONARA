import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC, NuSVC
from scipy import ndimage, fft
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
import os


'''Task2D'''
root_dir = r"C:\Users\loren\Documents\ComputationalAstrophysics\data_no_injection"

'''Task2E'''
#root_dir = r"C:\Users\loren\Documents\ComputationalAstrophysics\data_injected"

print(os.listdir(root_dir))

'''Data Preprocessor'''

class LightFluxProcessor:

    def __init__(self, fourier=True, normalize=True, gaussian=True, standardize=True):
        self.fourier = fourier
        self.normalize = normalize
        self.gaussian = gaussian
        self.standardize = standardize

    def fourier_transform(self, X):
        X = np.array(X)  # Ensure X is a NumPy array
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

        # Gaussian filter to smooth out data
        if self.gaussian:
            print("Applying Gaussian Filter...")
            df_train_x = ndimage.gaussian_filter(df_train_x, sigma=10)
            df_dev_x = ndimage.gaussian_filter(df_dev_x, sigma=10)

        if self.standardize:
            # Standardize X data
            print("Standardizing...")
            std_scaler = StandardScaler()
            df_train_x = std_scaler.fit_transform(df_train_x)
            df_dev_x = std_scaler.transform(df_dev_x)

        print("Finished Processing!")
        return df_train_x, df_dev_x


'''Load datasets'''

train_dataset_path = os.path.join(root_dir, "./exoTrain.csv")
dev_dataset_path = os.path.join(root_dir, "./exoTest.csv")

print("Loading datasets...")
df_train = pd.read_csv(train_dataset_path, encoding="ISO-8859-1")
df_dev = pd.read_csv(dev_dataset_path, encoding="ISO-8859-1")
print("Loaded datasets!")

# Reset index to ensure proper alignment
df_train = df_train.reset_index(drop=True)
df_dev = df_dev.reset_index(drop=True)

# Generate X and Y dataframe sets
df_train_x = df_train.drop('LABEL', axis=1)
df_dev_x = df_dev.drop('LABEL', axis=1)
df_train_y = df_train.LABEL
df_dev_y = df_dev.LABEL

# %matplotlib widget
data = df_train_x.to_numpy()
ii = 0
while ii < len(data):
    index = np.random.randint(0, len(data))
    label = df_train_y[index]
    if label == 2.0:
        plt.figure()
        plt.plot(range(len(data[index, :])), data[index, :])
        plt.title(f"Lightcurve at index {index} with label {label}")
        break
    ii += 1

'''Process data and create numpy matrices'''

def np_X_Y_from_df(df):
    df = shuffle(df)
    df_X = df.drop(['LABEL'], axis=1)
    X = np.array(df_X)
    Y_raw = np.array(df['LABEL']).reshape((len(df['LABEL']),1))
    Y = Y_raw == 2
    return X, Y

# Process dataset
LFP = LightFluxProcessor(
    fourier=True,
    normalize=True,
    gaussian=True,
    standardize=True)
df_train_x, df_dev_x = LFP.process(df_train_x, df_dev_x)

# Rejoin X and Y
df_train_processed = pd.DataFrame(df_train_x).join(pd.DataFrame(df_train_y))
df_dev_processed = pd.DataFrame(df_dev_x).join(pd.DataFrame(df_dev_y))

# Load X and Y numpy arrays
X_train, Y_train = np_X_Y_from_df(df_train_processed)
X_dev, Y_dev = np_X_Y_from_df(df_dev_processed)

'''Describe datasets'''

(num_examples, n_x) = X_train.shape # (n_x: input size, m : number of examples in the train set)
n_y = Y_train.shape[1] # n_y : output size
print("X_train.shape: ", X_train.shape)
print("Y_train.shape: ", Y_train.shape)
print("X_dev.shape: ", X_dev.shape)
print("Y_dev.shape: ", Y_dev.shape)
print("n_x: ", n_x)
print("num_examples: ", num_examples)
print("n_y: ", n_y)

'''Build, model, train and predict'''

#https://scikit-learn.org/1.5/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC Kernel initialization

kernels = ['linear', 'rbf', 'poly']  # Linear, Gaussian=rbf and Polynomial kernel
degree = 4  # Degree for the polynomial kernel
results = {}

for kernel in kernels:
    print(f"Training model with {kernel} kernel...")
    
    # Initialize model based on kernel
    if kernel == 'linear':
        model = LinearSVC()
    else:
        model = SVC(kernel=kernel, degree=degree if kernel == 'poly' else 3)  # Degree 4 for polynomial

    # Train the model
    model.fit(X_train, Y_train)
    
    # Predict the outputs
    train_outputs = model.predict(X_train)
    dev_outputs = model.predict(X_dev)

    # Round predictions to nearest integer (for classification)
    train_outputs = np.rint(train_outputs)
    dev_outputs = np.rint(dev_outputs)

    # Calculate metrics
    accuracy_train = accuracy_score(Y_train, train_outputs)
    accuracy_dev = accuracy_score(Y_dev, dev_outputs)
    precision_train = precision_score(Y_train, train_outputs)
    precision_dev = precision_score(Y_dev, dev_outputs)
    recall_train = recall_score(Y_train, train_outputs)
    recall_dev = recall_score(Y_dev, dev_outputs)
    confusion_matrix_train = confusion_matrix(Y_train, train_outputs)
    confusion_matrix_dev = confusion_matrix(Y_dev, dev_outputs)
    classification_report_train = classification_report(Y_train, train_outputs)
    classification_report_dev = classification_report(Y_dev, dev_outputs)

    # Store results for each kernel
    results[kernel] = {
        'accuracy_train': accuracy_train,
        'accuracy_dev': accuracy_dev,
        'precision_train': precision_train,
        'precision_dev': precision_dev,
        'recall_train': recall_train,
        'recall_dev': recall_dev,
        'confusion_matrix_train': confusion_matrix_train,
        'confusion_matrix_dev': confusion_matrix_dev,
        'classification_report_train': classification_report_train,
        'classification_report_dev': classification_report_dev
    }

    # Print metrics for each kernel
    print(f"\n{kernel} kernel metrics:\n")
    print("Train Set Error", 1.0 - accuracy_train)
    print("Dev Set Error", 1.0 - accuracy_dev)
    print("------------")
    print("Precision - Train Set", precision_train)
    print("Precision - Dev Set", precision_dev)
    print("------------")
    print("Recall - Train Set", recall_train)
    print("Recall - Dev Set", recall_dev)
    print("------------")
    print("Confusion Matrix - Train Set")
    print(confusion_matrix_train)
    print("Confusion Matrix - Dev Set")
    print(confusion_matrix_dev)
    print("------------")
    print("classification_report_train")
    print(classification_report_train)
    print("classification_report_dev")
    print(classification_report_dev)

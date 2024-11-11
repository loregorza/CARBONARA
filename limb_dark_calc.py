import numpy as np

def mean_columns(path):
    
    data = np.loadtxt(path, skiprows=2, usecols=[8,10])
    mean_c1 = np.mean(data[:, 0])
    mean_c2 = np.mean(data[:, 1])

    print(f"c1: {mean_c1}")
    print(f"c2: {mean_c2}")

mean_columns('ExoCTK_results.txt')
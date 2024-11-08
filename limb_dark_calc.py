import numpy as np
def mean_columns(file_path, col1, col2):
    # Load the data from the text file, assuming whitespace-separated values
    data = np.loadtxt(file_path)
    
    # Check if the file has enough columns
    if data.shape[1] < max(col1, col2):
        print(f"File does not have enough columns. It needs at least {max(col1, col2)} columns.")
        return

    # Calculate the mean of the specified columns
    mean_col1 = np.mean(data[:, col1 - 1])  # Subtract 1 because indexing starts from 0
    mean_col2 = np.mean(data[:, col2 - 1])

    print(f"Mean of column {col1}: {mean_col1}")
    print(f"Mean of column {col2}: {mean_col2}")

# Replace 'your_file.txt' with the path to your text file
mean_columns('ExoCTK_results.txt', 9, 11)
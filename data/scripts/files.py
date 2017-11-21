import numpy as np
import scipy.fftpack
from scipy.io import wavfile
from os import listdir
import random

thanos_path = 'files/first'
thanos_other_path = 'files/second'
test_path = 'files/test'

# Get a specific file.
def get_file_data(file_path):
    fs, data = wavfile.read(file_path)
    print('FFT: ', file_path)
    # data_normalized = [(i/2**8.) * 2-1 for i in data]
    # data_fft = fft(data_normalized)
    # return abs(data_fft[:(40000)])
    N = 2000
    data = scipy.fftpack.fft(data)
    data = np.abs(data[:N//2])
    data = 2*(data - np.max(data))/-np.ptp(data)-1
    return data

# Get all files from path.
def get_files_from_path(folder_path):
    files = []
    for filename in listdir(folder_path):
        try:
            files.append([get_file_data(folder_path + '/' + filename), filename])
        except:
            pass
    return files

# Get train files with labels.
def get_all_file_data():
    data = []
    
    # Get first path.
    for item in get_files_from_path(thanos_path):
        data.append([[1, 0], item[0]])

    # Get second path.
    for item in get_files_from_path(thanos_other_path):
        data.append([[0, 1], item[0]])
    
    # Shuffle files.
    random.shuffle(data)

    return data

# Get test files.
def get_test_files():
    return get_files_from_path(test_path)
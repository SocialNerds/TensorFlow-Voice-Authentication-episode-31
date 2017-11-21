import numpy as np
import scipy.fftpack
from scipy.io import wavfile
from os import listdir
import random
import sqlite3
import pickle

thanos_path = 'files/first'
thanos_other_path = 'files/second'
test_path = 'files/test'

# Get a specific file.
def get_file_data(file_path):
    # fs, data = wavfile.read(file_path)
    # print('FFT: ', file_path)
    # N = 2000
    # data = scipy.fftpack.fft(data)
    # data = np.abs(data[:N//2])
    # data = 2*(data - np.max(data))/-np.ptp(data)-1
    # return data
    table_name = 'fourier_data'
    database = 'data'
    # Connect to DB.
    conn = sqlite3.connect(table_name)
    c = conn.cursor()
    # Create table if it does not exist.
    sql = 'create table if not exists ' + table_name + ' (file_path text, data text)'
    c.execute(sql)
    conn.commit()
    
    c.execute('SELECT data FROM ' + table_name + ' WHERE file_path = ? ', (file_path,))
    data = c.fetchone()
    if data != None:
        conn.close()
        return pickle.loads(data)
    else:
        fs, data = wavfile.read(file_path)
        print('FFT: ', file_path)
        N = 2000
        data = scipy.fftpack.fft(data)
        data = np.abs(data[:N//2])
        data = 2*(data - np.max(data))/-np.ptp(data)-1

        # Save to DB.
        c.execute('insert into ' + table_name + '(file_path, data) values (?, ?)', (file_path, pickle.dumps(data, protocol=0)))
        conn.commit()
        conn.close()
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

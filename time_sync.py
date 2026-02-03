import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def time_sync(left_foot_path, right_foot_path, acc_path, gyro_path):
    left_foot = pd.read_csv(left_foot_path, delimiter = ';', dtype = float, skiprows=[1, 2, 3], decimal = ',')
    left_foot = left_foot.drop(columns=['Sensor nummer', 'Sync'])
    left_foot = left_foot.to_numpy()
    left_foot_mean = np.mean(left_foot, axis=1)
    left_max = left_foot_mean[:500].max()
    
    right_foot = pd.read_csv(right_foot_path, delimiter = ';', dtype = float, skiprows=[1, 2, 3], decimal = ',')
    right_foot = right_foot.drop(columns=['Sensor nummer', 'Sync'])
    right_foot = right_foot.to_numpy()
    right_foot_mean = np.mean(right_foot, axis=1)
    right_max = right_foot_mean[:500].max()

    accelerometer = pd.read_csv(acc_path, delimiter = ',', dtype = float, decimal = '.')
    acc_x = accelerometer['x'].to_numpy()
    acc_y = accelerometer['y'].to_numpy()
    acc_z = accelerometer['z'].to_numpy()


    gyroscope = pd.read_csv(gyro_path, delimiter = ',', dtype = float, decimal = '.')
    gyro_x = gyroscope['x'].to_numpy()
    gyro_y = gyroscope['y'].to_numpy()
    gyro_z = gyroscope['z'].to_numpy()

    

    if right_max >= left_max:
        foot_spike_index = np.where(right_foot_mean[:500] == right_foot_mean[:500].max())[0][0]
        print(foot_spike_index)
    else:
        foot_spike_index = np.where(left_foot_mean[:500] == left_foot_mean[:500].max())[0][0]
    
    print(acc_y[:500].max())
    print(np.where(acc_y[:500] == acc_y[:500].max())[0][0])
    acc_spike_index = np.where(acc_y[:500] == acc_y[:500].max())[0][0]
    if foot_spike_index < acc_spike_index:
        print('hej')
        acc_x = acc_x[acc_spike_index - foot_spike_index:]
        acc_y = acc_y[acc_spike_index - foot_spike_index:]
        acc_z = acc_z[acc_spike_index - foot_spike_index:]

        gyro_x = gyro_x[acc_spike_index - foot_spike_index:]
        gyro_y = gyro_y[acc_spike_index - foot_spike_index:]
        gyro_z = gyro_z[acc_spike_index - foot_spike_index:]

    elif foot_spike_index > acc_spike_index:
        right_foot = np.delete(right_foot, slice(0, foot_spike_index - acc_spike_index), axis=0)
        left_foot = np.delete(left_foot, slice(0, foot_spike_index - acc_spike_index), axis=0)

    if len(right_foot) < len(acc_x):
        acc_x = acc_x[:len(right_foot)]
        acc_y = acc_y[:len(right_foot)]
        acc_z = acc_z[:len(right_foot)]

        gyro_x = gyro_x[:len(right_foot)]
        gyro_y = gyro_y[:len(right_foot)]
        gyro_z = gyro_z[:len(right_foot)]

    elif len(right_foot) > len(acc_x):
        right_foot = right_foot[:len(acc_x)]
        left_foot = left_foot[:len(acc_x)]

    acc = np.column_stack((acc_x, acc_y, acc_z))
    gyro = np.column_stack((gyro_x, gyro_y, gyro_z))

    return right_foot, left_foot, acc, gyro

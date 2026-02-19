from interpolation import interpolate
from data_split_visualizer import pad_consistent_sensor_number
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


def comparison_time_average(left_foot, right_foot, cuts):
    left_foot = left_foot.drop(columns=['Sensor nummer', 'Sync'])
    left_foot_org = pad_consistent_sensor_number(left_foot.copy())[cuts[0]:cuts[1]]
    left_foot = interpolate(left_foot, len(left_foot.columns))[cuts[0]:cuts[1]]
    left_foot_mean = np.mean(left_foot_org, axis=2)

    right_foot = right_foot.drop(columns=['Sensor nummer', 'Sync'])
    right_foot_org = pad_consistent_sensor_number(right_foot.copy())[cuts[0]:cuts[1]]
    right_foot = interpolate(right_foot, len(right_foot.columns))[cuts[0]:cuts[1]]
    right_foot_mean = np.mean(right_foot_org, axis=2)
    return left_foot_mean, right_foot_mean



cuts = [[(500, -500), (500, -500)],
        [(500, -500), (500, -500)],
        [(500, -500), (500, -500)],
        [(500, -500), (1100, -500)],
        [(500, -1500), (500, -500)],
        [(500, -500), (500, -500)],
        [(500, -500), (500, -500)],
        [(1500, -500), (500, -500)],
        [(500, -500), (500, -500)],
        [(500, -500), (500, -500)]]

for id in tqdm(range(1,11)):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    left_feet = [pd.read_csv('../data_better/Subject' + str(id) + '/hand/subject'+ str(id) +'_hand_L.CSV', delimiter=';', dtype=float, skiprows=[1,2,3], decimal=","),
                pd.read_csv('../data_better/Subject' + str(id) + '/pocket/subject'+ str(id) +'_pocket_L.CSV', delimiter=';', dtype=float, skiprows=[1,2,3], decimal=",")]

    right_feet = [pd.read_csv('../data_better/Subject' + str(id) + '/hand/subject'+ str(id) +'_hand_R.CSV', delimiter=';', dtype=float, skiprows=[1,2,3], decimal=","),
                 pd.read_csv('../data_better/Subject' + str(id) + '/pocket/subject'+ str(id) +'_pocket_R.CSV', delimiter=';', dtype=float, skiprows=[1,2,3], decimal=",")]    
    
    for left, right, cut in zip(left_feet, right_feet, cuts[id-1]):
        left_foot_mean, right_foot_mean = comparison_time_average(left, right, cut)
        axes[0].plot(left_foot_mean, alpha = 0.6, label = f'Subject {id} l')
        axes[1].plot(right_foot_mean, alpha = 0.6, label = f'Subject {id} r')

axes[0].set_title("Left Foot - Comparison")
axes[1].set_title("Right Foot - Comparison")

axes[0].set_xlabel("Sensor ID")
axes[0].set_ylabel("Pressure")
axes[1].set_xlabel("Sensor ID")
axes[1].set_ylabel("Pressure")

axes[0].legend()
axes[1].legend()

plt.tight_layout()
plt.show()
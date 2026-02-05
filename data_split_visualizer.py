import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks, savgol_filter, medfilt, peak_prominences

def find_clearest_minima_heavy_noise(signal, n_minima=5,
                                     min_distance_ratio=0.1,
                                     prominence_factor=0.3,
                                     edge_exclude=1000):
    
    
    signal = np.asarray(signal, dtype=float)
    n = len(signal)

    #Median filter to remove spikes
    kernel_size = max(3, (n // 50) | 1)  # must be odd
    filtered = medfilt(signal, kernel_size=kernel_size)

    #Savitzky-Golay smoothing
    window = max(7, (n // 10) | 1)
    smoothed = savgol_filter(filtered, window_length=window, polyorder=3)

    #Noise-adaptive prominence
    signal_range = np.percentile(smoothed, 95) - np.percentile(smoothed, 5)
    min_prominence = prominence_factor * signal_range

    #Enforce minimum distance between minima
    min_distance = int(min_distance_ratio * n)
    

    minima_idx, _ = find_peaks(-smoothed, distance=min_distance)
    prominences = peak_prominences(-smoothed, minima_idx)[0]
    edge_mask = (minima_idx >= edge_exclude) & (minima_idx < n - edge_exclude)
    minima_idx = minima_idx[edge_mask]
    prominences = prominences[edge_mask]

    if len(minima_idx) == 0:
        return np.zeros(n, dtype=int)

    #Select top N minima by prominence
    order = np.argsort(prominences)[::-1][:n_minima]
    selected = minima_idx[order]

    #Create output mask
    binary_output = np.zeros(n, dtype=int)
    for i in range(200):
        binary_output[selected-100+i] = 1

    return binary_output

def find_crossing_indices(y1, y2):
    
    y1 = np.asarray(y1, dtype=float).ravel()
    y2 = np.asarray(y2, dtype=float).ravel()

    if y1.shape != y2.shape:
        raise ValueError("y1 and y2 must have the same shape")

    d = y1 - y2

    indices = []

    for i in range(len(d) - 1):
        d1, d2 = d[i], d[i + 1]

        # Exact hit
        if d1 == 0:
            indices.append(i)

        # True crossing
        elif d1 * d2 < 0:
            # pick the closer of i or i+1
            idx = i if abs(d1) < abs(d2) else i + 1
            indices.append(idx)

    return np.array(indices, dtype=int)

def pad_last_dim(tensors, value=-1):
    max_len = max(t.size(-1) for t in tensors)

    padded = [
        F.pad(t, (0, max_len - t.size(-1)), value=value)
        for t in tensors
    ]

    return padded

def pad_consistent_sensor_number(data, value=-1):
    desired_profile = ['29', '30', '31', 
                    '32', '33', '34', 
                    '40', '41', '42', 
                    '43', '44', '45', 
                    '46', '47', '51', 
                    '52', '53', '54', 
                    '55', '56', '57', 
                    '58', '59', '62', 
                    '63', '64', '65', 
                    '66', '67', '68', 
                    '69', '70', '71', 
                    '72', '74', '75', 
                    '76', '77', '78', 
                    '79', '80', '81', 
                    '82', '83', '84', 
                    '86', '87', '88', 
                    '89', '90', '91', 
                    '92', '93', '94', 
                    '95', '96', '98', 
                    '99', '100', '101', 
                    '102', '103', '104', 
                    '105', '106', '107', 
                    '110', '111', '112', 
                    '113', '114', '115', 
                    '116', '117', '118', 
                    '119', '122', '123', 
                    '124', '125', '126', 
                    '127', '128', '129', 
                    '130', '131', '135', 
                    '136', '137', '138', 
                    '139', '140', '141', 
                    '142', '143', '147', 
                    '148', '149', '150', 
                    '151', '152', '153', 
                    '154', '155', '159', 
                    '160', '161', '162', 
                    '163', '164', '165', 
                    '166', '167', '171', 
                    '172', '173', '174', 
                    '175', '176', '177', 
                    '178', '179', '183', 
                    '184', '185', '186', 
                    '187', '188', '189', 
                    '190', '191', '195', 
                    '196', '197', '198', 
                    '199', '200', '201', 
                    '202', '208', '209', 
                    '210', '211', '212', 
                    '213', '214', '221', 
                    '222', '223', '224', '225']
    i = 0
    while len(list(data)) <= len(desired_profile):
        if list(data)[i] != desired_profile[i]:
            data.insert(i, desired_profile[i], value)
        else:
            i += 1
    return data.to_numpy()

def process_subject(left_foot, right_foot):
    left_foot = left_foot.drop(columns=['Sensor nummer', 'Sync'])
    left_foot = pad_consistent_sensor_number(left_foot)[500:-500]
    left_foot_mean = np.mean(left_foot, axis=1)
    left_foot_average = np.mean(left_foot)

    right_foot = right_foot.drop(columns=['Sensor nummer', 'Sync'])
    right_foot = pad_consistent_sensor_number(right_foot)[500:-500]
    right_foot_mean = np.mean(right_foot, axis=1)
    right_foot_average = np.mean(right_foot)

    total = left_foot_mean + right_foot_mean
    total_average = np.mean(total)
    total = total-total_average
    total = total**2
    total_average = np.mean(total)

    indices = find_clearest_minima_heavy_noise(total)

    split_indicies = [x for x in range(len(indices)-1) if indices[x] != indices[x+1]]

    seq1_left = left_foot[0:split_indicies[0]]
    seq1_right = right_foot[0:split_indicies[0]]
    crossings_indicies_seq1 = find_crossing_indices(np.mean(seq1_left, axis=1), np.mean(seq1_right, axis=1))
    crossings_indicies_seq1_1 = crossings_indicies_seq1[::4]
    crossings_indicies_seq1_2 = crossings_indicies_seq1[1::4] 


    seq2_left = left_foot[split_indicies[1]:split_indicies[2]]
    seq2_right = right_foot[split_indicies[1]:split_indicies[2]]
    crossings_indicies_seq2 = find_crossing_indices(np.mean(seq2_left, axis=1), np.mean(seq2_right, axis=1))
    crossings_indicies_seq2_1 = crossings_indicies_seq2[::4]
    crossings_indicies_seq2_2 = crossings_indicies_seq2[1::4] 

    seq3_left = left_foot[split_indicies[3]:split_indicies[4]]
    seq3_right = right_foot[split_indicies[3]:split_indicies[4]]
    crossings_indicies_seq3 = find_crossing_indices(np.mean(seq3_left, axis=1), np.mean(seq3_right, axis=1))
    crossings_indicies_seq3_1 = crossings_indicies_seq3[::4]
    crossings_indicies_seq3_2 = crossings_indicies_seq3[1::4] 

    seq4_left = left_foot[split_indicies[5]:split_indicies[6]]
    seq4_right = right_foot[split_indicies[5]:split_indicies[6]]
    crossings_indicies_seq4 = find_crossing_indices(np.mean(seq4_left, axis=1), np.mean(seq4_right, axis=1))
    crossings_indicies_seq4_1 = crossings_indicies_seq4[::4]
    crossings_indicies_seq4_2 = crossings_indicies_seq4[1::4] 

    seq5_left = left_foot[split_indicies[7]:split_indicies[8]]
    seq5_right = right_foot[split_indicies[7]:split_indicies[8]]
    crossings_indicies_seq5 = find_crossing_indices(np.mean(seq5_left, axis=1), np.mean(seq5_right, axis=1))
    crossings_indicies_seq5_1 = crossings_indicies_seq5[::4]
    crossings_indicies_seq5_2 = crossings_indicies_seq5[1::4] 

    seq6_left = left_foot[split_indicies[9]:]
    seq6_right = right_foot[split_indicies[9]:]
    crossings_indicies_seq6 = find_crossing_indices(np.mean(seq6_left, axis=1), np.mean(seq6_right, axis=1))
    crossings_indicies_seq6_1 = crossings_indicies_seq6[::4]
    crossings_indicies_seq6_2 = crossings_indicies_seq6[1::4] 

    left_seqs = [seq1_left, seq2_left, seq3_left, seq4_left, seq5_left, seq6_left]
    right_seqs = [seq1_right, seq2_right, seq3_right, seq4_right, seq5_right, seq6_right]
    crossings_indicies_1 = [crossings_indicies_seq1_1, 
                            crossings_indicies_seq2_1, 
                            crossings_indicies_seq3_1, 
                            crossings_indicies_seq4_1, 
                            crossings_indicies_seq5_1, 
                            crossings_indicies_seq6_1]

    crossings_indicies_2 = [crossings_indicies_seq1_2, 
                            crossings_indicies_seq2_2, 
                            crossings_indicies_seq3_2, 
                            crossings_indicies_seq4_2, 
                            crossings_indicies_seq5_2, 
                            crossings_indicies_seq6_2]

    print(left_foot.shape)
    print(right_foot.shape)
    print(indices.shape)


    out_tensor = []
    for left, right, crossing1, crossing2 in zip(left_seqs, right_seqs, crossings_indicies_1, crossings_indicies_2):
        out_tensor_i = []
        for i in range(len(crossing1)-1):
            out_tensor_i.append(torch.concatenate([torch.Tensor(left[crossing1[i]:crossing1[i+1]]).permute([1,0]), 
                                                torch.Tensor(right[crossing1[i]:crossing1[i+1]]).permute([1,0])]))
        
        out_tensor_i = pad_last_dim(out_tensor_i)
        out_tensor_i = torch.stack(out_tensor_i)
        print(out_tensor_i.shape)
        out_tensor.append(out_tensor_i)
        out_tensor_j = []
        for j in range(len(crossing2)-1):
            out_tensor_j.append(torch.concatenate([torch.Tensor(left[crossing1[j]:crossing1[j+1]]).permute([1,0]), 
                                                torch.Tensor(right[crossing1[j]:crossing1[j+1]]).permute([1,0])]))
            
        out_tensor_j = pad_last_dim(out_tensor_j)
        out_tensor_j = torch.stack(out_tensor_j)
        print(out_tensor_j.shape)
        out_tensor.append(out_tensor_j)

    out_tensor = pad_last_dim(out_tensor)
    out_tensor = torch.concatenate(out_tensor, dim=0)
    print(out_tensor.shape)  

    fig = plt.figure(figsize=(10,7))
    subplot = fig.subfigures(2,1)
    ax1 = subplot[0].subplots(1)
    ax1.plot(left_foot_mean);
    ax1.plot(right_foot_mean);
    ax1.plot(indices*max([np.max(left_foot_mean), np.max(right_foot_mean)]));


    ax2 = subplot[1].subplots(3,2)
    ax2[0][0].plot(np.mean(seq1_left, axis=1))
    ax2[0][0].plot(np.mean(seq1_right, axis=1))
    ax2[0][0].scatter(crossings_indicies_seq1_1, np.mean(seq1_left)*np.ones_like(crossings_indicies_seq1_1), color='red')
    ax2[0][0].scatter(crossings_indicies_seq1_2, np.mean(seq1_left)*np.ones_like(crossings_indicies_seq1_2), color='green')

    ax2[1][0].plot(np.mean(seq2_left, axis=1))
    ax2[1][0].plot(np.mean(seq2_right, axis=1))
    ax2[1][0].scatter(crossings_indicies_seq2_1, np.mean(seq2_left)*np.ones_like(crossings_indicies_seq2_1), color='red')
    ax2[1][0].scatter(crossings_indicies_seq2_2, np.mean(seq2_left)*np.ones_like(crossings_indicies_seq2_2), color='green')

    ax2[2][0].plot(np.mean(seq3_left, axis=1))
    ax2[2][0].plot(np.mean(seq3_right, axis=1))
    ax2[2][0].scatter(crossings_indicies_seq3_1, np.mean(seq3_left)*np.ones_like(crossings_indicies_seq3_1), color='red')
    ax2[2][0].scatter(crossings_indicies_seq3_2, np.mean(seq3_left)*np.ones_like(crossings_indicies_seq3_2), color='green')

    ax2[0][1].plot(np.mean(seq4_left, axis=1))
    ax2[0][1].plot(np.mean(seq4_right, axis=1))
    ax2[0][1].scatter(crossings_indicies_seq4_1, np.mean(seq4_left)*np.ones_like(crossings_indicies_seq4_1), color='red')
    ax2[0][1].scatter(crossings_indicies_seq4_2, np.mean(seq4_left)*np.ones_like(crossings_indicies_seq4_2), color='green')

    ax2[1][1].plot(np.mean(seq5_left, axis=1))
    ax2[1][1].plot(np.mean(seq5_right, axis=1))
    ax2[1][1].scatter(crossings_indicies_seq5_1, np.mean(seq5_left)*np.ones_like(crossings_indicies_seq5_1), color='red')
    ax2[1][1].scatter(crossings_indicies_seq5_2, np.mean(seq5_left)*np.ones_like(crossings_indicies_seq5_2), color='green')

    ax2[2][1].plot(np.mean(seq6_left, axis=1))
    ax2[2][1].plot(np.mean(seq6_right, axis=1))
    ax2[2][1].scatter(crossings_indicies_seq6_1, np.mean(seq6_left)*np.ones_like(crossings_indicies_seq6_1), color='red')
    ax2[2][1].scatter(crossings_indicies_seq6_2, np.mean(seq6_left)*np.ones_like(crossings_indicies_seq6_2), color='green')

    plt.show()
    
    return out_tensor

left_feet = [pd.read_csv('../Subject5/hand/King_Megan_2026.02.02_11.49.47_L.CSV', delimiter=';', dtype=float, skiprows=[1,2,3], decimal=","),
             pd.read_csv('../Subject5/pocket/King_Megan_2026.02.02_11.46.01_L.CSV', delimiter=';', dtype=float, skiprows=[1,2,3], decimal=",")]

right_feet = [pd.read_csv('../Subject5/hand/King_Megan_2026.02.02_11.49.47_R.CSV', delimiter=';', dtype=float, skiprows=[1,2,3], decimal=","),
              pd.read_csv('../Subject5/pocket/King_Megan_2026.02.02_11.46.01_R.CSV', delimiter=';', dtype=float, skiprows=[1,2,3], decimal=",")]

samples = []

for left, right in zip(left_feet, right_feet):
    samples.append(process_subject(left, right))
samples = torch.concatenate(pad_last_dim(samples), dim=0)
torch.save(samples, '../data/subject5.pth')

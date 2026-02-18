import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
from interpolation import interpolate
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

import numpy as np

def find_crossing_indices(y1, y2):
    y1 = np.asarray(y1, dtype=float).ravel()
    y2 = np.asarray(y2, dtype=float).ravel()

    if y1.shape != y2.shape:
        raise ValueError("y1 and y2 must have the same shape")

    d = y1 - y2

    indices = []
    directions = []  # +1 upward, -1 downward

    for i in range(len(d) - 1):
        d1, d2 = d[i], d[i + 1]

        # Exact hit
        if d1 == 0:
            indices.append(i)

            # determine direction from next point if possible
            if d2 > 0:
                directions.append(+1)
            elif d2 < 0:
                directions.append(-1)
            else:
                directions.append(0)  # flat/indeterminate

        # True crossing
        elif d1 * d2 < 0:
            idx = i if abs(d1) < abs(d2) else i + 1
            indices.append(idx)

            # determine direction
            if d1 < 0 and d2 > 0:
                directions.append(+1)  # upward
            else:
                directions.append(-1)  # downward

    return np.array(indices, dtype=int), np.array(directions, dtype=int)

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
    while len(list(data)) < len(desired_profile):
        if i >= len(list(data)):
            data.insert(i, desired_profile[i], value)
        elif (list(data)[i] != desired_profile[i]):
            data.insert(i, desired_profile[i], value)
        else:
            i += 1
    return data.to_numpy()

def process_subject(left_foot, right_foot, cuts):
    
    left_foot = left_foot.drop(columns=['Sensor nummer', 'Sync'])
    left_foot_org = pad_consistent_sensor_number(left_foot.copy())[cuts[0]:cuts[1]]
    left_foot = interpolate(left_foot, len(left_foot.columns))[cuts[0]:cuts[1]]
    left_foot_mean = np.mean(left_foot_org, axis=1)
    left_foot_average = np.mean(left_foot)

    right_foot = right_foot.drop(columns=['Sensor nummer', 'Sync'])
    right_foot_org = pad_consistent_sensor_number(right_foot.copy())[cuts[0]:cuts[1]]
    right_foot = interpolate(right_foot, len(right_foot.columns))[cuts[0]:cuts[1]]
    right_foot_mean = np.mean(right_foot_org, axis=1)
    right_foot_average = np.mean(right_foot)
    
    """left_foot = left_foot.drop(columns=['Sensor nummer', 'Sync'])
    left_foot = pad_consistent_sensor_number(left_foot.copy())[cuts[0]:cuts[1]]
    left_foot_mean = np.mean(left_foot, axis=1)
    left_foot_average = np.mean(left_foot)

    right_foot = right_foot.drop(columns=['Sensor nummer', 'Sync'])
    right_foot = pad_consistent_sensor_number(right_foot.copy())[cuts[0]:cuts[1]]
    right_foot_mean = np.mean(right_foot, axis=1)
    right_foot_average = np.mean(right_foot)"""

    total = left_foot_mean + right_foot_mean
    total_average = np.mean(total)
    total = total-total_average
    total = total**2
    total_average = np.mean(total)

    indices = find_clearest_minima_heavy_noise(total)
    print(indices)

    split_indicies = [x for x in range(len(indices)-1) if indices[x] != indices[x+1]]
    print(split_indicies)

    
    seq1_left = left_foot_org[0:split_indicies[0]]
    seq1_right = right_foot_org[0:split_indicies[0]]
    crossings_indicies_seq1 = find_crossing_indices(np.mean(seq1_left, axis=1), np.mean(seq1_right, axis=1))
    crossings_indicies_seq1 = crossings_indicies_seq1[0][::4] if crossings_indicies_seq1[1][0] == 1 else crossings_indicies_seq1[0][1::4]

    seq2_left = left_foot_org[split_indicies[1]:split_indicies[2]]
    seq2_right = right_foot_org[split_indicies[1]:split_indicies[2]]
    crossings_indicies_seq2 = find_crossing_indices(np.mean(seq2_left, axis=1), np.mean(seq2_right, axis=1))
    crossings_indicies_seq2 = crossings_indicies_seq2[0][1::4] if crossings_indicies_seq2[1][0] == 1 else crossings_indicies_seq2[0][::4]

    seq3_left = left_foot_org[split_indicies[3]:split_indicies[4]]
    seq3_right = right_foot_org[split_indicies[3]:split_indicies[4]]
    crossings_indicies_seq3 = find_crossing_indices(np.mean(seq3_left, axis=1), np.mean(seq3_right, axis=1))
    crossings_indicies_seq3 = crossings_indicies_seq3[0][::4] if crossings_indicies_seq3[1][0] == 1 else crossings_indicies_seq3[0][1::4]

    seq4_left = left_foot_org[split_indicies[5]:split_indicies[6]]
    seq4_right = right_foot_org[split_indicies[5]:split_indicies[6]]
    crossings_indicies_seq4 = find_crossing_indices(np.mean(seq4_left, axis=1), np.mean(seq4_right, axis=1))
    crossings_indicies_seq4 = crossings_indicies_seq4[0][1::4] if crossings_indicies_seq4[1][0] == 1 else crossings_indicies_seq4[0][::4]

    seq5_left = left_foot_org[split_indicies[7]:split_indicies[8]]
    seq5_right = right_foot_org[split_indicies[7]:split_indicies[8]]
    crossings_indicies_seq5 = find_crossing_indices(np.mean(seq5_left, axis=1), np.mean(seq5_right, axis=1))
    crossings_indicies_seq5 = crossings_indicies_seq5[0][::4] if crossings_indicies_seq5[1][0] == 1 else crossings_indicies_seq5[0][1::4]

    seq6_left = left_foot_org[split_indicies[9]:]
    seq6_right = right_foot_org[split_indicies[9]:]
    crossings_indicies_seq6 = find_crossing_indices(np.mean(seq6_left, axis=1), np.mean(seq6_right, axis=1))
    crossings_indicies_seq6 = crossings_indicies_seq6[0][1::4] if crossings_indicies_seq6[1][0] == 1 else crossings_indicies_seq6[0][::4]
    
    
    
    seq1_left_interpolated = left_foot[0:split_indicies[0]]
    seq1_right_interpolated = right_foot[0:split_indicies[0]]

    seq2_left_interpolated = left_foot[split_indicies[1]:split_indicies[2]]
    seq2_right_interpolated = right_foot[split_indicies[1]:split_indicies[2]]

    seq3_left_interpolated = left_foot[split_indicies[3]:split_indicies[4]]
    seq3_right_interpolated = right_foot[split_indicies[3]:split_indicies[4]]

    seq4_left_interpolated = left_foot[split_indicies[5]:split_indicies[6]]
    seq4_right_interpolated = right_foot[split_indicies[5]:split_indicies[6]]

    seq5_left_interpolated = left_foot[split_indicies[7]:split_indicies[8]]
    seq5_right_interpolated = right_foot[split_indicies[7]:split_indicies[8]]

    seq6_left_interpolated = left_foot[split_indicies[9]:]
    seq6_right_interpolated = right_foot[split_indicies[9]:]

    left_seqs = [seq1_left, seq2_left, seq3_left, seq4_left, seq5_left, seq6_left]
    right_seqs = [seq1_right, seq2_right, seq3_right, seq4_right, seq5_right, seq6_right]
    
    left_seqs_interpolated = [seq1_left_interpolated, 
                              seq2_left_interpolated, 
                              seq3_left_interpolated, 
                              seq4_left_interpolated, 
                              seq5_left_interpolated, 
                              seq6_left_interpolated]
    right_seqs_interpolated = [seq1_right_interpolated, 
                               seq2_right_interpolated, 
                               seq3_right_interpolated, 
                               seq4_right_interpolated, 
                               seq5_right_interpolated, 
                               seq6_right_interpolated]
    
    crossings_indicies_1 = [crossings_indicies_seq1, 
                            crossings_indicies_seq2, 
                            crossings_indicies_seq3, 
                            crossings_indicies_seq4, 
                            crossings_indicies_seq5, 
                            crossings_indicies_seq6]

    print(left_foot.shape)
    print(right_foot.shape)
    print(left_foot_org.shape)
    print(right_foot_org.shape)
    print(indices.shape)


    out_tensor_standard = []
    out_tensor_interpolated = []
    for left, right, crossing1 in zip(left_seqs, right_seqs, crossings_indicies_1):
        out_tensor_i = []
        for i in range(len(crossing1)-1):
            out_tensor_i.append(torch.concatenate([torch.Tensor(left[crossing1[i]:crossing1[i+1]]).permute([1,0]), 
                                                torch.Tensor(right[crossing1[i]:crossing1[i+1]]).permute([1,0])]))
        
        out_tensor_i = pad_last_dim(out_tensor_i)
        out_tensor_i = torch.stack(out_tensor_i)            
        out_tensor_standard.append(out_tensor_i)
    
    
    for left, right, crossing1 in zip(left_seqs_interpolated, right_seqs_interpolated, crossings_indicies_1):
        out_tensor_i = []
        for i in range(len(crossing1)-1):
            out_tensor_i.append(torch.concatenate([torch.Tensor(left[crossing1[i]:crossing1[i+1]]).permute([1,0]), 
                                                torch.Tensor(right[crossing1[i]:crossing1[i+1]]).permute([1,0])]))
        
        out_tensor_i = pad_last_dim(out_tensor_i)
        out_tensor_i = torch.stack(out_tensor_i)
        out_tensor_interpolated.append(out_tensor_i)






    out_tensor_standard = pad_last_dim(out_tensor_standard)
    out_tensor_standard = torch.concatenate(out_tensor_standard, dim=0)
    out_tensor_interpolated = pad_last_dim(out_tensor_interpolated)
    out_tensor_interpolated = torch.concatenate(out_tensor_interpolated, dim=0)
    
    print(out_tensor_standard.shape)  
    print(out_tensor_interpolated.shape)  

    """fig = plt.figure(figsize=(10,7))
    subplot = fig.subfigures(2,1)
    ax1 = subplot[0].subplots(1)
    ax1.plot(left_foot_mean);
    ax1.plot(right_foot_mean);
    #ax1.plot(indices*max([np.max(left_foot_mean), np.max(right_foot_mean)]));


    ax2 = subplot[1].subplots(3,2)
    ax2[0][0].plot(np.mean(seq1_left, axis=1))
    ax2[0][0].plot(np.mean(seq1_right, axis=1))
    ax2[0][0].scatter(crossings_indicies_seq1, np.mean(seq1_left)*np.ones_like(crossings_indicies_seq1), color='red')

    ax2[1][0].plot(np.mean(seq2_left, axis=1))
    ax2[1][0].plot(np.mean(seq2_right, axis=1))
    ax2[1][0].scatter(crossings_indicies_seq2, np.mean(seq2_left)*np.ones_like(crossings_indicies_seq2), color='red')

    ax2[2][0].plot(np.mean(seq3_left, axis=1))
    ax2[2][0].plot(np.mean(seq3_right, axis=1))
    ax2[2][0].scatter(crossings_indicies_seq3, np.mean(seq3_left)*np.ones_like(crossings_indicies_seq3), color='red')

    ax2[0][1].plot(np.mean(seq4_left, axis=1))
    ax2[0][1].plot(np.mean(seq4_right, axis=1))
    ax2[0][1].scatter(crossings_indicies_seq4, np.mean(seq4_left)*np.ones_like(crossings_indicies_seq4), color='red')

    ax2[1][1].plot(np.mean(seq5_left, axis=1))
    ax2[1][1].plot(np.mean(seq5_right, axis=1))
    ax2[1][1].scatter(crossings_indicies_seq5, np.mean(seq5_left)*np.ones_like(crossings_indicies_seq5), color='red')

    ax2[2][1].plot(np.mean(seq6_left, axis=1))
    ax2[2][1].plot(np.mean(seq6_right, axis=1))
    ax2[2][1].scatter(crossings_indicies_seq6, np.mean(seq6_left)*np.ones_like(crossings_indicies_seq6), color='red')

    plt.show()"""
    
    return out_tensor_standard, out_tensor_interpolated, F.normalize(out_tensor_interpolated.detach().clone(), dim=1)


if __name__ == "__main__":
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
        left_feet = [pd.read_csv('../data_better/Subject' + str(id) + '/hand/subject'+ str(id) +'_hand_L.CSV', delimiter=';', dtype=float, skiprows=[1,2,3], decimal=","),
                    pd.read_csv('../data_better/Subject' + str(id) + '/pocket/subject'+ str(id) +'_pocket_L.CSV', delimiter=';', dtype=float, skiprows=[1,2,3], decimal=",")]

        right_feet = [pd.read_csv('../data_better/Subject' + str(id) + '/hand/subject'+ str(id) +'_hand_R.CSV', delimiter=';', dtype=float, skiprows=[1,2,3], decimal=","),
                    pd.read_csv('../data_better/Subject' + str(id) + '/pocket/subject'+ str(id) +'_pocket_R.CSV', delimiter=';', dtype=float, skiprows=[1,2,3], decimal=",")]

        samples = []
        samples_interpolated = []
        samples_normalized = [] 
        for left, right, cut in zip(left_feet, right_feet, cuts[id-1]):
            out = process_subject(left, right, cut)
            samples.append(out[0])
            samples_interpolated.append(out[1])
            samples_normalized.append(out[2])
        samples = torch.concatenate(pad_last_dim(samples), dim=0)
        samples_interpolated = torch.concatenate(pad_last_dim(samples_interpolated), dim=0)
        samples_normalized = torch.concatenate(pad_last_dim(samples_normalized), dim=0)
        torch.save(samples, '../data/subject'+str(id)+'.pth')
        torch.save(samples_interpolated, '../data_interpolate/subject'+str(id)+'.pth')
        torch.save(samples_normalized, '../data_normalized/subject'+str(id)+'.pth')

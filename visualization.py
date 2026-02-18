from dataset import dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import torch.nn.functional as F
import torch
path_list = ['../data_interpolate/subject1.pth', '../data_interpolate/subject2.pth', '../data_interpolate/subject3.pth', '../data_interpolate/subject4.pth',
              '../data_interpolate/subject5.pth', '../data_interpolate/subject6.pth', '../data_interpolate/subject7.pth', '../data_interpolate/subject8.pth', 
              '../data_interpolate/subject9.pth', '../data_interpolate/subject10.pth']
data = dataset(path_list)
samples = np.array(data.samples)
padding_size = samples.shape[-1]
labels = data.labels
n_samples, n_channels, n_time = samples.shape
samples_reshaped = samples.reshape(n_samples, n_channels * n_time)
scaler = StandardScaler().fit(samples_reshaped)
reducer = umap.UMAP().fit(scaler.transform(samples_reshaped))
counter = 1
plt.figure(figsize=(10,7))
for path in path_list:
    subject = dataset([path])
    samples = subject.samples
    for i in range(len(samples)):
        samples[i] = F.pad(samples[i], (0, padding_size - samples[i].shape[1]), value=-1)
    samples = np.array(samples)
    n_samples, n_channels, n_time = samples.shape
    samples_reshaped = samples.reshape(n_samples, n_channels * n_time)
    scaled = scaler.transform(samples_reshaped)
    samples_umap = reducer.transform(scaled)
    plt.scatter(
            samples_umap[:, 0], 
            samples_umap[:, 1],
            alpha=0.6,
            label=f'Class {counter}'
    )
    counter += 1

plt.legend()
plt.title("UMAP Projection - Multiple Subjects")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.show()


# l vs r
"""counter = 1
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for path in path_list:
    subject = dataset([path])
    samples = subject.samples
    for i in range(len(samples)):
        samples[i] = F.pad(samples[i], (0, padding_size - samples[i].shape[1]), value=-1)
    samples = np.array(samples)
    samples_l = samples.copy()
    samples_l[:, 151:, :] = 0
    samples_r = samples.copy()
    samples_r[:, :151, :] = 0
    n_samples, n_channels, n_time = samples.shape
    samples_reshaped = samples.reshape(n_samples, n_channels * n_time)
    n_samples = samples.shape[0]
    samples_l_reshaped = samples_l.reshape(n_samples, -1)
    samples_r_reshaped = samples_r.reshape(n_samples, -1)
    scaled = scaler.transform(samples_reshaped)
    scaled_l = scaler.transform(samples_l_reshaped)
    scaled_r = scaler.transform(samples_r_reshaped)
    samples_umap = reducer.transform(scaled)
    samples_l_umap = reducer.transform(scaled_l)
    samples_r_umap = reducer.transform(scaled_r)
    axes[0].scatter(
        samples_l_umap[:, 0],
        samples_l_umap[:, 1],
        alpha=0.6,
        label=f'Subject {counter}'
    )

    axes[1].scatter(
        samples_r_umap[:, 0],
        samples_r_umap[:, 1],
        alpha=0.6,
        label=f'Subject {counter}'
    )
    counter += 1

axes[0].set_title("Left Foot - UMAP")
axes[1].set_title("Right Foot - UMAP")

axes[0].set_xlabel("UMAP 1")
axes[0].set_ylabel("UMAP 2")
axes[1].set_xlabel("UMAP 1")

axes[0].legend()
axes[1].legend()

plt.tight_layout()
plt.show()"""
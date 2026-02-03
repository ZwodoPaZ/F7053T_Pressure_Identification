import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, w):
        return  np.convolve(data, np.ones(w), 'valid') / w
    

left_foot = pd.read_csv('../Subject2/pocket/Fahlen_Oscar_2026.02.02_11.03.54_L.CSV', delimiter=';', dtype=float, skiprows=[1,2,3], decimal=",")
left_foot = left_foot.drop(columns=['Sensor nummer', 'Sync'])
left_foot = left_foot.to_numpy()
left_foot = np.mean(left_foot, axis=1)
left_foot_average = np.mean(left_foot)

right_foot = pd.read_csv('../Subject2/pocket/Fahlen_Oscar_2026.02.02_11.03.54_R.CSV', delimiter=';', dtype=float, skiprows=[1,2,3], decimal=",")
right_foot = right_foot.drop(columns=['Sensor nummer', 'Sync'])
right_foot = right_foot.to_numpy()
right_foot = np.mean(right_foot, axis=1)
right_foot_average = np.mean(right_foot)

total = left_foot + right_foot
total_average = np.mean(total)
total = total-total_average
total = total**2
#total = moving_average(total, 400)
#total_average = np.mean(total)

#left_foot = left_foot[0:len(total)]
#right_foot = right_foot[0:len(total)]

print(left_foot.shape)
print(right_foot.shape)

plt.figure(figsize=(10,7))
plt.subplot(4,1,1)
#plt.plot([x*0.01 for x in range(len(left_foot))],left_foot);
plt.plot(left_foot);
plt.plot([left_foot_average for _ in range(len(left_foot))]);
plt.subplot(4,1,2)
#plt.plot([x*0.01 for x in range(len(right_foot))], right_foot);
plt.plot(right_foot);
plt.plot([right_foot_average for _ in range(len(right_foot))])
plt.subplot(4,1,3)
plt.plot(left_foot);
plt.plot(right_foot);
plt.subplot(4,1,4)
plt.plot(total)
plt.plot([0.75*total_average for _ in range(len(total))])
plt.show()
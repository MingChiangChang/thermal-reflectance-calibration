import yaml

import numpy as np
import matplotlib.pyplot as plt

DW = 6637
PW = 49

data = []

for i in range(20):
    d = np.load(f"../data/npy/{DW}us_{PW}W_{i}_pfit.npy")
    data.append((d[5:-5,0]+d[5:-5,-1]).tolist())

data = np.array(data)
data[data>700] = np.mean(data)
data[data<100] = np.mean(data)
upper = 4*np.std(data)
mean = np.mean(data)
data[data > mean + upper] = np.mean(data)
data[data < mean - upper] = np.mean(data)
data[data>700] = np.mean(data)
data[data<100] = np.mean(data)
print(f"""Mean: {np.mean(data)}
          Std:  {np.std(data)}
          Max:  {np.max(data)}
          Min:  {np.min(data)}""")
x, y = np.indices(data.shape)

#sc = plt.scatter(x, y, c=data, vmin=400, vmax=600)
sc = plt.imshow(data.T, vmin=400, vmax=650, aspect=0.75, extent=(-38, 38, -17, 17))
plt.colorbar(sc)
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.title("Peak temperature at each position")
plt.show()


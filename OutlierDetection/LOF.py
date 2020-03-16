from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (10.0, 5.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

rng = np.random.RandomState(42)
n_samples=6  #样本总数

dataset = pd.read_csv('../csv/Water Quality Record.csv', header=0, index_col=0, parse_dates=True)
data = dataset.values.reshape(-1)

values = dataset.values
groups = [0, 1, 2, 3]
# fig, axs = plt.subplots(1)

df = pd.DataFrame(dataset)  # 整体数据的全部字典类型
do = df['Dissolved Oxygen']  # 返回溶解氧那一列，用字典的方式

DO = []
for i in range(0, len(do)):
    DO.append([do[i]])
DO = np.array(DO)

# fit the model
clf = LocalOutlierFactor(n_neighbors=20,contamination=0.09)

DO_copy = DO
m = 0
pre = clf.fit_predict(DO)
for i in range(len(pre)):
    if pre[i] == -1:
        plt.scatter(i,DO[i],c='r',linewidths=1)
        DO_copy = np.delete(DO_copy,i-m,0)
        print(i)
        m+=1


plt.plot(DO)
plt.text(0,2,str(m),ha='left',c='r')
plt.show()
print("\n")
print(m)
print("\n")
print(len(DO_copy))
plt.plot(DO_copy)
plt.show()
from sklearn.cluster import DBSCAN
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

df = pd.DataFrame(dataset)  # 整体数据的全部字典类型
do = df['Dissolved Oxygen']  # 返回溶解氧那一列，用字典的方式

DO = []
for i in range(0, len(do)):
    DO.append([do[i]])
DO = np.array(DO)

eps, min_samples = 0.1, 20
# eps为领域的大小，min_samples为领域内最小点的个数
model = DBSCAN(eps=eps, min_samples=min_samples) # 构造分类器
pre = model.fit_predict(DO)

m = 0
plt.plot(DO)
for i in range(len(pre)):
    if pre[i] == -1:
        plt.scatter(i,DO[i],c='r',linewidths=1)
        m+=1

plt.text(2,2,str(m),c='r')
plt.show()





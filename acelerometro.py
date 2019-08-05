import pandas as pd
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, fft

nomes = ['Amab', 'Raf', 'Die']

for nome in nomes:
    accel = pd.read_csv('Accelerometer_'+nome+'.csv', sep='\t', header=None).drop(labels=[0, 1, 2], axis=1)
    print(accel.head())
    accel.to_csv('accel.csv', header=None, index=None)

    f = open('accel.csv')
    data = f.read()
    f.close()
    data = data.replace('[', '')
    data = data.replace(']', '')
    data = data.replace('"', '')
    data = data.splitlines()

    accel = []

    for linhas in data:
        linhas = linhas.split(',')
        valores = []
        for valor in linhas:
            valores.append(float(valor))
        # euler = tf.euler_from_quaternion(valores, 'rxyz')
        q = [valores[1], valores[2], valores[3], valores[0]]
        euler = R.from_quat(q).as_euler('xyz', degrees=True)
        valores.append(euler[0])
        valores.append(euler[1])
        valores.append(euler[2])

        accel.append(valores)

    accelerometer = pd.DataFrame(accel, columns=['angulo', 'x', 'y', 'z', 'rX', 'rY', 'rZ'])

    accelerometer.to_csv('accel_'+nome+'.csv', index=False)

# Interpolando data
for nome in nomes:
    data = pd.read_csv('accel_' + nome + '.csv')
    colunas = ['rX', 'rY', 'rZ']
    x = np.arange(0, data.shape[0])
    # xN = np.arange(0, data.shape[0], 100)
    #
    # data_interp = pd.DataFrame(columns=colunas)

    # for col in colunas:
    #     y = data[col]
    #     F = interpolate.interp1d(x, y)
    #     data_interp[col] = F(xN)

    for coluna in colunas:
        plt.figure(dpi=300)
        plt.plot((x/100), data[coluna])
        # plt.scatter(data_interp.index, data_interp[coluna])
        plt.title(coluna.capitalize())
        plt.xlabel("Tempo (s)")
        plt.ylabel("Graus")
        plt.grid(True)
        plt.savefig(coluna + '_' + nome + '.png')
        plt.show()


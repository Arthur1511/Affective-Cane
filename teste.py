import pandas as pd
from scipy.spatial.transform import Rotation as R

accel = pd.read_csv('dados/Accelerometer_Amab.csv', sep='\t', header=None).drop(labels=[0, 1, 2], axis=1)
print(accel.head())
accel.to_csv('dados/accel.csv', header=None, index=None)

f = open('dados/accel.csv')
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

accelerometer.to_csv('dados/accel2.csv')

fsr = pd.read_csv('dados/FSR_Data', sep='\t').drop(columns=['Id', 'Day', 'Unnamed: 2', 'Unnamed: 4', 'Value'])
fsr = fsr.rename({'Time': "fsr"}, axis=1)

print(accelerometer.describe())
print(fsr.describe())

dataframe = pd.concat([fsr, accelerometer], axis=1)
print(dataframe.describe())

dataframe.to_csv('dados/Dataframe.csv', index=None)

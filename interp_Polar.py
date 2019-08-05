from matplotlib import axis
from scipy import interpolate, fft
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import hrvanalysis


# Interpolação dos dados do Polar de segundo a segundo
polar = pd.read_csv("Polar_Data.csv", sep=';', header=None)

for i in range(np.shape(polar)[0]):
    polar.iloc[i, 0] = 0

for i in range(np.shape(polar)[0]):
    if i == 0:
        polar.iloc[i, 0] = polar.iloc[i, 1]

    polar.iloc[i, 0] = polar.iloc[i - 1, 0] + polar.iloc[i, 1]

F = interpolate.interp1d(polar[0], polar[1], kind='linear')

xN = np.arange(1000, polar[0].max(), 1000)
y = F(xN).round()
nPolar = pd.DataFrame({'tempo': xN, 'polar': y, 'bpm': (60000 / y).round()})
nPolar['tempo'] = nPolar['tempo'] / 1000
polar[0] = polar[0] / 1000

plt.figure(dpi=300)
plt.plot(nPolar['tempo'], nPolar['bpm'])
plt.title('Frequência Cardíaca')
plt.xlabel('tempo (s)')
plt.ylabel('BPM')
plt.savefig('bpm.png')
plt.grid(True)
plt.show()
# plt.plot(nPolar['tempo'], nPolar['polar'], '-')
# plt.show()

plt.figure(dpi=300)
plt.boxplot(nPolar['bpm'])
plt.title('Boxplot Frequência Cardíaca')
plt.ylabel('BPM')
plt.savefig('bpm_boxplot.png')
plt.show()


# fsr = data['fsr']
# x = np.arange(0, fsr.size)
# xN = np.arange(0, fsr.size, 100)
# F_fsr = interpolate.interp1d(x, fsr)
# nFsr = F_fsr(xN)
# n_Fsr = pd.DataFrame({'tempo': xN / 100, 'valor': nFsr})
#
# plt.plot(n_Fsr['tempo'], n_Fsr['valor'], '-')
# plt.show()

# Interpolando data
data = pd.read_csv("Dataframe.csv")
colunas = data.columns
x = np.arange(0, data.shape[0])
xN = np.arange(0, data.shape[0], 100)

data_interp = pd.DataFrame(columns=colunas)

for col in colunas:
    y = data[col]
    F = interpolate.interp1d(x, y)
    data_interp[col] = F(xN)

data_interp = pd.concat([data_interp, nPolar.drop(axis=1, columns=['tempo', 'bpm'])], axis=1)

for coluna in data.columns:
    plt.figure(dpi=300)
    plt.plot(x/100, data[coluna])
    # plt.scatter(data_interp.index, data_interp[coluna])
    plt.title(coluna.upper())
    plt.xlabel("Tempo (s)")
    plt.grid(True)
    # plt.savefig(coluna + '.png')
    plt.show()

# FFT do FSR interpolado
# tf = fft(data_interp['fsr'])
# T = 1 / 100
# N = len(data_interp['fsr'])
# x = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
#
# # plt.subplot(2, 1, 2)
# plt.plot(x, 2.0 / N * np.abs(tf[:N // 2]))
# plt.xlim(0, 50)
# plt.xlabel("Hz")
# plt.ylabel("FFT")
# plt.title("FFT FSR interpolado")
# plt.show()

# FFT dataset interpolado
for coluna in data.columns:
    tf = fft(data_interp[coluna].dropna())
    if coluna == 'polar':
        T = 1
    else:
        T = 1 / 100

    N = len(data_interp[coluna].dropna())
    x = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

    # plt.subplot(2, 1, 2)
    plt.figure(dpi=300)
    plt.plot(x, 2.0 / N * np.abs(tf[:N // 2]))
    # plt.xlim(0, 50)
    plt.grid(True)
    plt.xlabel("Hz")
    plt.ylabel("FFT")
    plt.title("FFT " + coluna + " interpolado")
    plt.savefig(coluna + '_FFT.png')
    plt.show()

# FFT dataset bruto
for coluna in data.columns:
    tf = fft(data[coluna] - data[coluna].mean())
    # tf[0] = 0

    T = 1 / 100

    N = len(data[coluna].dropna())
    x = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

    # plt.subplot(2, 1, 2)
    plt.figure(dpi=300)
    plt.plot(x, 2.0 / N * np.abs(tf[:N // 2]))
    plt.xlim(0, 10)
    plt.grid(True)
    plt.xlabel("Hz")
    plt.ylabel("FFT")
    plt.title("FFT " + coluna)
    plt.savefig(coluna + '_Bruto_FFT.png')
    plt.show()

# fft do polar
polar = nPolar.drop(['tempo', 'bpm'], axis=1)
tf = fft(polar - polar.mean())
T = 1
N = len(polar)
x = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

plt.figure(dpi=300)
plt.subplot(2, 1, 1)
plt.plot(x, 2.0 / N * np.abs(tf[:N // 2]))
plt.scatter(x, 2.0 / N * np.abs(tf[:N // 2]))
plt.vlines([0.04, 0.15, 0.4], ymin=-0.5, ymax=8, linestyles='dashed', colors=['b', 'y', 'r'], label=['VLF LF HF'])
# plt.subplot(3, 1, 3)
# plt.specgram(polar, Fs=1, detrend='mean', scale='dB', mode='magnitude', xextent=(0, 1))
# plt.xlim(0, 1)
plt.xlabel("Hz")
plt.ylabel("FFT")
plt.title("FFT Polar")
plt.subplot(2, 1, 2)
plt.plot(polar)
plt.grid()
# plt.savefig("Polar_Bruto_FFT.png")
plt.show()

corr = data_interp.corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, cmap="RdBu", annot=True, robust=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()


plt.figure(figsize=(7, 15))
colunas = ['rX', 'rY', 'rZ']
for i, col in enumerate(colunas):
    plt.subplot(3, 1, i + 1)
    # plt.plot(np.arange(0, 64, 0.01), (data[col]*180)/math.pi)
    plt.boxplot(data[col])
    plt.legend()
    plt.xlabel("Tempo (s)")
    plt.ylabel("Graus")
    plt.grid()

plt.show()

# plotly.offline.plot({
#     "data": [go.Scatter(x=x, y=(data['rY'] * 180) / math.pi)],
#     "layout": go.Layout(title="Polar FFT")
# }, auto_open=True)

data_interp = data_interp.diff()

data_interp = data_interp.dropna(axis=0)

corr = data_interp.corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, cmap="RdBu", annot=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

plt.title("Correlation Heatmap", fontsize=24, color="darkred")
plt.show()







import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import transformations

plt.style.use('ggplot')  # if you are an R user and want to feel at home

caneLog = pd.read_csv('Dataframe.csv')

print(caneLog.head())



# data = pd.DataFrame(columns=['Dfsr', 'Dangle', 'Dx', 'Dy', 'Dz'])
# data['Dfsr'] = caneLog['fsr'].diff()
# data['Dangle'] = caneLog['angulo'].diff()
# data['Dx'] = caneLog['x'].diff()
# data['Dy'] = caneLog['y'].diff()
# data['Dz'] = caneLog['z'].diff()
# data = data.dropna(axis=0)
# print(data.head())
# print(data.describe())
#
# # Correlation between fsr output and angulo
#
# data.plot(x='Dfsr', y='Dangle', kind="scatter",
#              figsize=[10, 10],
#              color="b", alpha=0.3,
#              fontsize=14)
# plt.title("Delta FSR vs Delta Angulo",
#           fontsize=24, color="darkred")
#
# corr = data.corr()
# plt.figure(figsize=(9, 7))
# sns.heatmap(corr, cmap="RdBu", robust=True, annot=True,
#             xticklabels=corr.columns.values,
#             yticklabels=corr.columns.values)
#
# plt.title("Correlation Heatmap", fontsize=24, color="darkred")
# plt.show()

# caneLog.plot(x='fsr', y='angulo', kind="scatter",
#              figsize=[10, 10],
#              color="b", alpha=0.3,
#              fontsize=14)
# plt.title("FSR vs Angulo",
#           fontsize=24, color="darkred")
#
# plt.xlabel("FSR", fontsize=18)
#
# plt.ylabel("Angulo", fontsize=18)
#
# plt.show()
#
# # Correlation between Angulo and X
#
# caneLog.plot(x='x', y='angulo', kind="scatter", figsize=[10, 10], color="g", alpha=0.3, fontsize=14)
#
# plt.title("Angulo vs X", fontsize=24, color="darkred")
#
# plt.xlabel("X", fontsize=18)
#
# plt.ylabel("Angulo", fontsize=18)
#
# plt.show()
#
# # Correlation between Angulo and Y
#
# caneLog.plot(x='y', y='angulo', kind="scatter", figsize=[10, 10], color="y", alpha=0.3, fontsize=14)
#
# plt.title("Angulo vs Y", fontsize=24, color="darkred")
#
# plt.xlabel("Y", fontsize=18)
#
# plt.ylabel("Angulo", fontsize=18)
#
# plt.show()
#
# # Correlation between Angulo and Z
#
# caneLog.plot(x='z', y='angulo', kind="scatter", figsize=[10, 10], color="b", alpha=0.3, fontsize=14)
#
# plt.title("Angulo vs Z", fontsize=24, color="darkred")
#
# plt.xlabel("Z", fontsize=18)
#
# plt.ylabel("Angulo", fontsize=18)
#
# plt.show()
#
#
# # correlation heatmap
#
# corr = caneLog.corr()
# plt.figure(figsize=(9, 7))
# sns.heatmap(corr, cmap="RdBu", annot=True,
#             xticklabels=corr.columns.values,
#             yticklabels=corr.columns.values)
#
# plt.title("Correlation Heatmap", fontsize=24, color="darkred")
# plt.show()

#%%
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error 
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV, GroupKFold

#%%
"""Training Data"""

# Reading, getting RUL # 
col_names = ['unit_nr','life_cycles','os_1','os_2','os_3']
for i in range(1,27):
    col_names.append(f'sns_{i}')
df_train = pd.read_csv('FD001/train_FD001.txt',names=col_names,sep=r'\s+')
df_train.drop(columns = ['sns_22','sns_23','sns_24','sns_25','sns_26'],inplace=True)

df_train_max_cycles = df_train.groupby('unit_nr')['life_cycles'].max().reset_index()
df_train_max_cycles.columns = ['unit_nr','max_cycles']

df_train = df_train.merge(df_train_max_cycles,on='unit_nr')
df_train['RUL'] = df_train['max_cycles'] - df_train['life_cycles']
df_train.drop(columns='max_cycles',inplace=True)

# Checking average correlations of sensor readings with corresponding RUL across engines #
per_engine_mean = df_train.groupby('unit_nr').apply(lambda x: x.corr(method='spearman')['RUL'])
# sns_1, sns_5, sns_10, sns_16, sns_18, sns_19 to be discarded due to constant value for all RULs. No variation.
# sns_6 and sns_14's correlation had to be investigated through graph. sns_14 had good correlation near last half of engine life.
# sns_6 showed less correlation graphically and can be discarded. 

df_train.drop(columns=['sns_1','sns_5','sns_10','sns_16','sns_18','sns_19'],inplace=True)

""" Data Transformation """

df_train_input = df_train.drop(columns=['unit_nr','life_cycles','os_1','os_2','os_3'])
df_train_label = df_train['RUL'].copy().clip(upper=125)

# Scaling # 
scaler = StandardScaler()
df_train_input = scaler.fit_transform(df_train_input)

# Adding Rate of Change 
#%%
plt.figure(figsize=(12,9))
sns.lineplot(data=df_train[(df_train['RUL']<800) & (df_train['unit_nr'].isin(np.arange(1,20)))],x='RUL',y='sns_19',hue='unit_nr')
plt.show()





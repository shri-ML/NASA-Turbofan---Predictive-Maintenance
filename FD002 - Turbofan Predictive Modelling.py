#%% Modules Import
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score,GroupKFold

from xgboost import XGBRegressor

#%% Operations on Training Data
col_names = ['unit_nr','life_cycles','os_1','os_2','os_3','os_4','os_5','os_6']
for i in range(1,19):
    col_names.append(f'sns_{i}')
df_train = pd.read_csv('FD002/train_FD002.txt',names=col_names,sep=r'\s+')

df_corr = df_train.groupby('unit_nr').apply(lambda x: x.drop(columns=['unit_nr']).corr(method='spearman')['life_cycles']).reset_index()
df_corr



#%% Training Models on Transformed Training Data

#%% Checking Model Performance on Testing Data

#%% Summarizing

#%% Plotting Space
plt.figure(figsize=(20,10))
sns.lineplot(data=df_train[df_train['unit_nr']==1],x='life_cycles',y='os_2')
plt.xticks(np.arange(0,200,50))
plt.show()
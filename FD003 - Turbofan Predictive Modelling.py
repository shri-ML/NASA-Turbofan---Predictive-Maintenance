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

#%%
# Getting initial data # 
col_names = ['unit_nr','time_cycles','os_1','os_2','os_3']
for i in range(1,27):
    col_names.append(f'sns_{i}')
df = pd.read_csv('FD003/train_FD003.txt', names=col_names, sep=r'\s+')
df = df.drop(columns=['sns_22','sns_23','sns_24','sns_25','sns_26'])
df[col_names[5:26]].describe()

# Inserting RUL # 
max_cycles = df.groupby('unit_nr')[['time_cycles']].max().reset_index()
max_cycles.columns = ['unit_nr','time_cycles1']
df = df.merge(max_cycles,on='unit_nr')
df.insert(loc=2,column='RUL',value=df['time_cycles1']-df['time_cycles'])
df = df.drop(columns='time_cycles1')

#%%
# Checking sensor correlation w.r.t RUL by plotting a graph # 
df_1 = df[df['unit_nr']==1]
df_1.info()
plt.figure(figsize=(12,9))
sns.lineplot(data=df,x=df['RUL'],y=df['sns_15'])
plt.show()

#%%
# Checking sensor correlation w.r.t RUL by using Spearman Correlation Score # 
cols_to_drop = ['unit_nr','time_cycles','os_1','os_2','os_3']
df_2 = df.drop(columns=cols_to_drop)
df_2.corr(method='spearman')['RUL'].sort_values()

# Selecting and removing the selected columns # 
cols_to_drop_1 = ['sns_1','sns_5','sns_16','sns_18','sns_19']
df_dropped = df_2.drop(columns=cols_to_drop_1)
df_dropped

# Final Table with only correlated columns and initial operating parameters #
df_corr_only = pd.concat([df[cols_to_drop[0:2]],df_dropped],axis=1)
core = df_corr_only.corr(method='spearman') # Final Correlations 
print(core['RUL'].sort_values(ascending=False))

# Splitting Data into X & Y # 
df_train_input = df_corr_only.drop(columns=['RUL','unit_nr','time_cycles'])
df_train_label = df_corr_only['RUL'].copy()

# Scaling, Clipping Data # 
scale = StandardScaler()
df_train_input = pd.DataFrame(scale.fit_transform(df_train_input),columns=df_train_input.columns)
df_train_label = df_train_label.clip(upper=125)

#%%
# Predictive Modelling with Linear Regression & Random Forest #
linr = LinearRegression()
linr.fit(df_train_input,df_train_label)
df_train_predict_linr = linr.predict(df_train_input)
rmse1 = root_mean_squared_error(df_train_label,df_train_predict_linr)
print('Training RMSE for Linear Regression: ',rmse1)

forest = RandomForestRegressor(max_features='sqrt')
forest.fit(df_train_input,df_train_label)
df_train_predict_rfr = forest.predict(df_train_input)
rmse2 = root_mean_squared_error(df_train_label,df_train_predict_rfr)
print('Training RMSE for Random Forest:',rmse2)

#%%
# Test Data: Reading, Filtering Columns, Scaling, Clipping.. # 
df_test = pd.read_csv('FD003/test_FD003.txt',names=col_names,sep=r'\s+')
df_test = df_test.drop(columns=['sns_22','sns_23','sns_24','sns_25','sns_26'])

df_RUL = pd.read_csv('FD003/RUL_FD003.txt',names=['RUL'])
df_RUL['unit_nr'] = df_RUL.index+1
df_RUL['max_cycles'] = df_test.groupby('unit_nr').last().reset_index()['time_cycles'] + df_RUL['RUL']
df_RUL.drop(columns='RUL')

df_test = df_test.merge(df_RUL,on='unit_nr')
df_test['RUL'] = df_test['max_cycles'] - df_test['time_cycles']

cols_to_drop_test = cols_to_drop_1 + cols_to_drop + ['RUL','max_cycles']
df_test_input = df_test.drop(columns=cols_to_drop_test)
df_test_label = df_test['RUL'].copy()

# Scaling Data #
scale_test = StandardScaler()
df_test_input = pd.DataFrame(scale.transform(df_test_input),columns=df_train_input.columns)
df_test_label = df_test_label.clip(upper=125)

# Checking RMSE for Test Data #
rmse_test_lr = root_mean_squared_error(df_test_label,linr.predict(df_test_input))
rmse_test_rf = root_mean_squared_error(df_test_label,forest.predict(df_test_input))
print('Testing RMSE, Linear Regression: ',rmse_test_lr)
print('Testing RMSE, Random Forest: ',rmse_test_rf)



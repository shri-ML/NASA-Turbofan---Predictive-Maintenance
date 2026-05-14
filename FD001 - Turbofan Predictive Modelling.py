#%%
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error 
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score

from xgboost import XGBRegressor

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

""" Adding δ of sns_4 values to check if it has any contribution to RUL """

df_train['roll_mean_sns_11'] = df_train.groupby('unit_nr')['sns_11'].transform(lambda x: x.rolling(window=5,min_periods=1).mean())
df_train['roll_mean_sns_4'] = df_train.groupby('unit_nr')['sns_4'].transform(lambda x: x.rolling(window=5,min_periods=1).mean())

""" Data Transformation """

df_train_input = df_train.drop(columns=['unit_nr','life_cycles','os_1','os_2','os_3','RUL'])
df_train_label = df_train['RUL'].copy().clip(upper=125)

# Scaling # 
scaler = StandardScaler()
df_train_input = pd.DataFrame(data=scaler.fit_transform(df_train_input),columns=df_train_input.columns)

#%%
""" Fitting some models and checking their Prediction RMSE """

groups = GroupKFold(n_splits=5)

lr = LinearRegression()
lr.fit(df_train_input,df_train_label)
rmse_lr = root_mean_squared_error(df_train_label,lr.predict(df_train_input))
print('LR RMSE: ',rmse_lr)

rfr = RandomForestRegressor(max_features='sqrt', n_jobs=-1,random_state=42,max_depth=13)
rfr.fit(df_train_input,df_train_label)
rmse_rfr = root_mean_squared_error(df_train_label,rfr.predict(df_train_input))
print('RFR RMSE: ',rmse_rfr)
cross_vrfr = pd.Series(-cross_val_score(estimator=rfr,X=df_train_input,y=df_train_label,cv=groups,groups=df_train['unit_nr'],scoring='neg_root_mean_squared_error'))
print('RFR CV Mean: ',cross_vrfr.mean())

gdb = GradientBoostingRegressor(random_state=42)
gdb.fit(df_train_input,df_train_label)
rmse_gdb = root_mean_squared_error(df_train_label,gdb.predict(df_train_input))
print('GDB RMSE: ',rmse_gdb)
cross_vgdb = pd.Series(-cross_val_score(estimator=gdb,X=df_train_input,y=df_train_label,cv=groups,
                                        groups=df_train['unit_nr'],scoring='neg_root_mean_squared_error'))
print('GDB CV Mean: ',cross_vgdb.mean())

xgb = XGBRegressor(max_depth = 3)
xgb.fit(df_train_input,df_train_label)
rmse_xgb = root_mean_squared_error(df_train_label,xgb.predict(df_train_input))
print('XGB RMSE: ',rmse_xgb)
cross_xgb = pd.Series(-cross_val_score(estimator=xgb, X=df_train_input,y=df_train_label,cv=groups,groups=df_train['unit_nr'],scoring='neg_root_mean_squared_error'))
print('XGB CV Mean: ',cross_xgb.mean())


#%%
""" Test Data """
df_test = pd.read_csv('FD001/test_FD001.txt',names=col_names,sep=r'\s+')
df_test.drop(columns=['sns_22','sns_23','sns_24','sns_25','sns_26'],inplace=True)

df_test_RUL_read = pd.read_csv('FD001/RUL_FD001.txt',names=['RUL_Last'])
df_test_RUL_read['unit_nr'] = df_test_RUL_read.index+1
df_test_RUL_read['life_cycles_corresp'] = df_test.groupby('unit_nr')['life_cycles'].last().values
df_test_RUL_read['max_cycles'] = df_test_RUL_read['life_cycles_corresp']+df_test_RUL_read['RUL_Last']

df_test = df_test.merge(df_test_RUL_read[['unit_nr','max_cycles']],on='unit_nr')
df_test['RUL'] = df_test['max_cycles'] - df_test['life_cycles']
df_test.drop(columns=['sns_1','sns_5','sns_10','sns_16','sns_18','sns_19','max_cycles'],inplace=True)

df_test['roll_mean_sns_11'] = df_test.groupby('unit_nr')['sns_11'].transform(lambda x: x.rolling(window=5,min_periods=1).mean())
df_test['roll_mean_sns_4']  = df_test.groupby('unit_nr')['sns_4'].transform(lambda x: x.rolling(window=5,min_periods=1).mean())

# Scaling Input Features # 
df_test_input = df_test.drop(columns=['unit_nr','life_cycles','os_1','os_2','os_3','RUL'])
df_test_input = pd.DataFrame(scaler.transform(df_test_input),columns=df_test_input.columns)
df_test_label = df_test['RUL'].copy().clip(upper=125)

rmse_test_rfr = root_mean_squared_error(df_test_label,rfr.predict(df_test_input))
print('Test Error, RFR: ',rmse_test_rfr)

rmse_test_xgb = root_mean_squared_error(df_test_label,xgb.predict(df_test_input))
print('Test Error, XGB: ',rmse_test_xgb)


















#%%: Adding Rate of Change 
path = rfr.estimators_[0].cost_complexity_pruning_path(df_train_input,df_train_label)
imp_plot_df = pd.DataFrame({'ccp_alpha':path.ccp_alphas,'impurity':path.impurities})

plt.figure(figsize=(12,9))
sns.lineplot(data=imp_plot_df,x='ccp_alpha',y='impurity')
plt.xlim(0,100)
plt.xticks(np.arange(0,100,25))
plt.show()

plt.figure(figsize=(12,9))
sns.lineplot(x=df_train.index[:200],y=df_train.iloc[:200]['sns_14'])
plt.xticks(np.arange(0,200,25))
plt.show()

plt.figure(figsize=(12,9))
sns.lineplot(data=df_train[(df_train['RUL']<800) & (df_train['unit_nr'].isin(np.arange(1,20)))],x='RUL',y='sns_19',hue='unit_nr')
plt.show()





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

#%% Operations on Training & Testing Data
# Data extraction #
col_names = ['unit_nr','life_cycles','os_1','os_2','os_3','os_4','os_5','os_6']
col_names_1 = col_names.copy()
for i in range(1,19):
    col_names.append(f'sns_{i}')
df_train = pd.read_csv('FD002/train_FD002.txt',names=col_names,sep=r'\s+')
df_test  = pd.read_csv('FD002/test_FD002.txt',names=col_names,sep=r'\s+')

# Creating an operating condition combination and grouping by that combination # 
df_train['os_1'] = df_train['os_1'].round(0)
df_train['os_2'] = df_train['os_2'].round(1)
df_train['os_combination'] = df_train[['os_1','os_2','os_3']].astype(str).agg('-'.join,axis=1)

df_test['os_1'] = df_test['os_1'].round(0)
df_test['os_2'] = df_test['os_2'].round(1)
df_test['os_combination'] = df_test[['os_1','os_2','os_3']].astype(str).agg('-'.join,axis=1)

# Sensor values are to be transformed to normal distribution. Transformation is avoided on some columns # 
col_names_1 = col_names_1 + ['sns_13','sns_15','sns_16']
col_names_2 = [x for x in col_names if x not in col_names_1]

# Select columns are transformed. Mean and std are stored for use in next column # 
means = df_train.groupby('os_combination')[col_names_2].mean() 
stds  = df_train.groupby('os_combination')[col_names_2].std()
df_train_filtered = df_train.drop(columns=col_names_1).groupby('os_combination').transform(lambda x: (x - x.mean())/x.std())
df_train[col_names_2] = df_train_filtered[col_names_2]

# Test Data columns have to be transformed w.r.t df_train mean and std #  
df_test = df_test.merge(means.add_suffix('_mean').reset_index(), on='os_combination')
df_test = df_test.merge(stds.add_suffix('_std').reset_index(), on='os_combination')
for i in col_names_2:
    df_test[i] = (df_test[i] - df_test[f'{i}_mean'])/df_test[f'{i}_std']
df_test.drop(columns=[f'{x}_mean' for x in col_names_2]+[f'{x}_std' for x in col_names_2],inplace=True)

# RUL is added #
df_train_RUL = df_train.groupby('unit_nr')['life_cycles'].max().reset_index()
df_train_RUL.columns = ['unit_nr','max_cycles']
df_train = df_train.merge(df_train_RUL,on='unit_nr')
df_train['RUL'] = df_train['max_cycles'] - df_train['life_cycles']
df_train['RUL%'] = 100*(df_train['RUL']/df_train['max_cycles'])
df_train.drop(columns='max_cycles',inplace=True)

df_test_RUL = pd.read_csv('FD002/RUL_FD002.txt',names=['max_RUL'],sep=r'\s+')
df_test_RUL['unit_nr'] = df_test_RUL.index + 1
df_test_RUL['max_cycles'] = df_test.groupby('unit_nr')['life_cycles'].last().reset_index(drop=True)
df_test_RUL['max_life'] = df_test_RUL['max_cycles']+df_test_RUL['max_RUL']
df_test = df_test.merge(df_test_RUL[['unit_nr','max_life']],on='unit_nr')
df_test['RUL'] = df_test['max_life'] - df_test['life_cycles'] 
df_test.drop(columns='max_life',inplace=True)

# RUL is clipped to 125 # 
df_train['RUL'] = df_train['RUL'].clip(upper=125)
df_test['RUL'] = df_test['RUL'].clip(upper=125)

# Spearman Correlation on RUL% (Training Data) # 
df_train.drop(columns=['os_combination','RUL','life_cycles','unit_nr']).corr(method='spearman')['RUL%'].sort_values(ascending=False)

# Columns with good correlation are only kept # 
cols_keep = ['sns_6','sns_5','sns_10',
             'sns_11','sns_9_rollmean','sns_14_rollmean','sns_12_rollmean',
             'sns_1_rollmean','sns_4_rollmean','sns_8_rollmean','sns_17_rollmean','sns_18_rollmean']

# Rolling Average is added # 
df_train['sns_9_rollmean'] = df_train.groupby('unit_nr')['sns_9'].transform(lambda x: x.rolling(window = 5,min_periods=1).mean())
df_train['sns_14_rollmean'] = df_train.groupby('unit_nr')['sns_14'].transform(lambda x: x.rolling(window = 5,min_periods=1).mean())
df_train['sns_12_rollmean'] = df_train.groupby('unit_nr')['sns_12'].transform(lambda x: x.rolling(window = 5,min_periods=1).mean())
df_train['sns_1_rollmean'] = df_train.groupby('unit_nr')['sns_1'].transform(lambda x: x.rolling(window = 5,min_periods=1).mean())
df_train['sns_8_rollmean'] = df_train.groupby('unit_nr')['sns_8'].transform(lambda x: x.rolling(window = 5,min_periods=1).mean())
df_train['sns_4_rollmean'] = df_train.groupby('unit_nr')['sns_4'].transform(lambda x: x.rolling(window = 5,min_periods=1).mean())
df_train['sns_17_rollmean'] = df_train.groupby('unit_nr')['sns_17'].transform(lambda x: x.rolling(window = 5,min_periods=1).mean())
df_train['sns_18_rollmean'] = df_train.groupby('unit_nr')['sns_18'].transform(lambda x: x.rolling(window = 5,min_periods=1).mean())

df_test['sns_9_rollmean'] = df_test.groupby('unit_nr')['sns_9'].transform(lambda x: x.rolling(window = 5,min_periods=1).mean())
df_test['sns_14_rollmean'] = df_test.groupby('unit_nr')['sns_14'].transform(lambda x: x.rolling(window = 5,min_periods=1).mean())
df_test['sns_12_rollmean'] = df_test.groupby('unit_nr')['sns_12'].transform(lambda x: x.rolling(window = 5,min_periods=1).mean())
df_test['sns_1_rollmean'] = df_test.groupby('unit_nr')['sns_1'].transform(lambda x: x.rolling(window = 5,min_periods=1).mean())
df_test['sns_8_rollmean'] = df_test.groupby('unit_nr')['sns_8'].transform(lambda x: x.rolling(window = 5,min_periods=1).mean())
df_test['sns_4_rollmean'] = df_test.groupby('unit_nr')['sns_4'].transform(lambda x: x.rolling(window = 5,min_periods=1).mean())
df_test['sns_17_rollmean'] = df_test.groupby('unit_nr')['sns_17'].transform(lambda x: x.rolling(window = 5,min_periods=1).mean())
df_test['sns_18_rollmean'] = df_test.groupby('unit_nr')['sns_18'].transform(lambda x: x.rolling(window = 5,min_periods=1).mean())

# Inputs, Labels  
df_train_input = df_train[cols_keep].copy()
df_train_labels = df_train['RUL'].copy()

df_test_input = df_test[cols_keep].copy()
df_test_labels = df_test['RUL'].copy()

#%% Training Models on Transformed Training Data
# KFolds Group for Cross Validation # 
groups = GroupKFold(n_splits=5)

# Linear Regression #
linreg = LinearRegression()
linreg.fit(df_train_input,df_train_labels)
rmse_lr = root_mean_squared_error(df_train_labels,linreg.predict(df_train_input))
print('LR RMSE: ',rmse_lr)

# Random Forest # 
rf = RandomForestRegressor(random_state=42,max_features='sqrt',n_jobs=-1,max_depth = 15)
rf.fit(df_train_input,df_train_labels)
rmse_rf = root_mean_squared_error(df_train_labels,rf.predict(df_train_input))
print('RF RMSE: ',rmse_rf)
rf_cv = -cross_val_score(estimator=rf,X=df_train_input,y=df_train_labels,cv=groups,groups=df_train['unit_nr'],scoring='neg_root_mean_squared_error',n_jobs=-1)
print(rf_cv.mean())

# XGBoost # 
xgb = XGBRegressor(learning_rate=0.05,reg_alpha = 0.1)
xgb.fit(df_train_input,df_train_labels)
rmse_xgb = root_mean_squared_error(df_train_labels,rf.predict(df_train_input))
print('XGB RMSE: ',rmse_xgb)
xgb_cv = -cross_val_score(estimator=xgb,X=df_train_input,y=df_train_labels,cv=groups,groups=df_train['unit_nr'],scoring='neg_root_mean_squared_error',n_jobs=-1)
print(xgb_cv.mean())

#%% Checking Model Performance on Testing Data
rmse_test_lr = root_mean_squared_error(df_test_labels,linreg.predict(df_test_input))
print('Test LR RMSE: ',rmse_test_lr)

rmse_test_rf = root_mean_squared_error(df_test_labels,rf.predict(df_test_input))
print('Test RF RMSE: ',rmse_test_rf)

rmse_test_xgb = root_mean_squared_error(df_test_labels,xgb.predict(df_test_input))
print('Test XGB RMSE: ',rmse_test_xgb)

#%% Summarizing
# 1. Learned to use .agg(). Used to classify different operating conditions
# 2. Used Rolling for Mean(). Used more columns this time for greater accuracy.
# 3. Presentation is cleaner.

#%% Plotting Space
plt.figure(figsize=(20,10))
sns.lineplot(data=df_train[df_train['unit_nr']==1],x='life_cycles',y='os_2')
plt.xticks(np.arange(0,200,50))
plt.show()
#%% Modules Import 
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

#%% - Filtering Training Data
""" Getting initial data """
col_names = ['unit_nr','time_cycles','os_1','os_2','os_3']
for i in range(1,27):
    col_names.append(f'sns_{i}')
df_read_train = pd.read_csv('FD003/train_FD003.txt', names=col_names, sep=r'\s+')
df_read_train = df_read_train.drop(columns=['sns_22','sns_23','sns_24','sns_25','sns_26'])

# Inserting RUL # 
max_cycles = df_read_train.groupby('unit_nr')[['time_cycles']].max().reset_index()
max_cycles.columns = ['unit_nr','time_cycles1']
df_read_train = df_read_train.merge(max_cycles,on='unit_nr')
df_read_train.insert(loc=2,column='RUL',value=df_read_train['time_cycles1']-df_read_train['time_cycles'])
df_read_train = df_read_train.drop(columns='time_cycles1')

""" Filtering Training Data: Checking correlation scores, etc """
# Checking sensor correlation w.r.t RUL by using Spearman Correlation Score # 
df_read_train.corr(method='spearman')['RUL'].sort_values()

# Creating the final table by removing columns with less correlation and initial_setting columns # 
cols_to_drop = ['unit_nr','time_cycles','os_1','os_2','os_3'] # Initial Setting Columns
cols_to_drop_1 = ['sns_1','sns_5','sns_16','sns_18','sns_19']
df_dropped = df_read_train.drop(columns=cols_to_drop_1+cols_to_drop)
df_corr_only = pd.concat([df_read_train[cols_to_drop[0:2]],df_dropped],axis=1)

# Splitting Data into X & Y # 
df_train_input = df_corr_only.drop(columns=['RUL','unit_nr','time_cycles'])
df_train_label = df_corr_only['RUL'].copy()

# Scaling, Clipping Data # 
scale = StandardScaler()
df_train_input = pd.DataFrame(scale.fit_transform(df_train_input),columns=df_train_input.columns)
df_train_label = df_train_label.clip(upper=125)

#%% - Linear Regression & Random Forest, tuning hyperparameters
""" Predictive Modelling with Linear Regression & Random Forest """
linr = LinearRegression()
linr.fit(df_train_input,df_train_label)
df_train_predict_linr = linr.predict(df_train_input)
rmse1 = root_mean_squared_error(df_train_label,df_train_predict_linr)
print('Training RMSE for Linear Regression: ',rmse1)

forest = RandomForestRegressor(random_state=42)
forest.fit(df_train_input,df_train_label)
df_train_predict_rfr = forest.predict(df_train_input)
rmse2 = root_mean_squared_error(df_train_label,df_train_predict_rfr)
print('Training RMSE for Random Forest:',rmse2)

"""K-Folds Cross Validation to check max_depth hyperparameter"""
folds = GroupKFold(n_splits=5)
set_params = {'max_depth':[8,10,11,12,15],'max_features':['sqrt',None]}
grids = GridSearchCV(forest,set_params,cv=folds,scoring='neg_root_mean_squared_error')
grids.fit(df_train_input,df_train_label,groups=df_corr_only['unit_nr'])
print(pd.DataFrame(grids.cv_results_)) # Result: max_depth = 12, max_features = 'sqrt' #

#%% - Test Data Verification
""" Test Data: Reading, Filtering Columns, Scaling, Clipping.. """
forest_final = grids.best_estimator_

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
df_test_input = pd.DataFrame(scale.transform(df_test_input),columns=df_train_input.columns)
df_test_label = df_test_label.clip(upper=125)

# Checking RMSE for Test Data #
rmse_test_lr = root_mean_squared_error(df_test_label,linr.predict(df_test_input))
rmse_test_rf = root_mean_squared_error(df_test_label,forest_final.predict(df_test_input))
print('Testing RMSE, Linear Regression: ',rmse_test_lr)
print('Testing RMSE, Random Forest: ',rmse_test_rf)

#%% - Results & Inference/Learnings

# Results: Testing RMSE, Random Forest:  14.297771016864115. max_depth = 12, max_features = 'sqrt'
   
# 1. Hyperparameters Tuning: More Clarity on CCP (Cost Complexity Pruning), max_depth, minimum impurity decrease, max_features
# 2. Plotting Trees (using plot tree function from sklearn.tree)
# 3. Spearman Correlation for Curved Data
# 4. Group K-Folds in Cross-Validation
# 5. Remember! np.where, pd.cut(..)..

#%% - Plotting Space

""" Plotting Space"""

"""Plotting Impurity v/s ccp_alpha"""
path = forest.estimators_[0].cost_complexity_pruning_path(df_train_input,df_train_label)
imp_plot_df = pd.DataFrame({'alpha':path.ccp_alphas,'impurity':path.impurities})

plt.figure(figsize=(12,9))
sns.lineplot(data=imp_plot_df,x='alpha',y='impurity')
plt.xlim(0,10)
plt.xticks(np.arange(0,10,1))
plt.show()

"""Plotting Impurity for each node"""
impure = forest.estimators_[0].tree_
dfn = pd.DataFrame({'impurity':impure.impurity,'weighted_n':impure.weighted_n_node_samples,
                         'left':impure.children_left,'right':impure.children_right})
                         
dfn_splits = dfn[dfn['left']!=-1].copy()
left_idx = dfn_splits['left']
right_idx = dfn_splits['right']

left_impurity = (dfn['weighted_n'][left_idx].values/dfn_splits['weighted_n'].values)*dfn['impurity'][left_idx].values
right_impurity = (dfn['weighted_n'][right_idx].values/dfn_splits['weighted_n'].values)*dfn['impurity'][right_idx].values

dfn_splits['impurity decrease'] = (dfn_splits['weighted_n'].values/24720)*(dfn_splits['impurity'].values-left_impurity - right_impurity)

plt.figure(figsize=(12,9))
sns.histplot(x=dfn_splits['impurity decrease'],bins=50)
plt.xlim(0, 3)
plt.show()




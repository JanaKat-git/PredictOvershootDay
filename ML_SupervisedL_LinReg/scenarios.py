'''
Predict OvershootDay for differenr sceanrios. For a scenario where all features change with 
the same yearly decreas use line 15-20. If only one feature changes and all the other features
follow the trend use line 24-29 of this script.
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from model_functions import prepare_df, make_dateofyear, model_gradientboosting, predict_OD, calculate_scenario_all, calculate_scenario_feature

#Create df for prediction (generate data)
df_OD = prepare_df('./../data/deficit/world_ef_bc_tot.csv', 'world_perpared_total.csv', 'total')
ef = df_OD.drop(index=[1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969])

df_predictions = calculate_scenario_all(ef, 0.98,  2031, 13)
plt.figure(figsize=(10,8))
plt.plot(df_predictions.index[-13:], df_predictions[['cropLand']].iloc[-13:])
plt.xlabel('years', fontdict={'fontsize': 15})
plt.ylabel('*10⁹ global hectars', fontdict={'fontsize': 15})
plt.title('Development of the CropLand-footprint by an yearly increas of 2 % from 2017')


# df_predictions = calculate_scenario_feature('./data_csv/df_01_full.csv', 2017, 0.99,  'carbon_pred')
# plt.figure(figsize=(10,8))
# plt.plot(df_predictions.index[-13:], df_predictions[['carbon_pred']].iloc[-13:])
# plt.xlabel('years', fontdict={'fontsize': 15})
# plt.ylabel('*10⁹ global hectars', fontdict={'fontsize': 15})
# plt.title('Development of the Carbonfootprint by an yearly increas of 2 % from 2017')


#Prediction
X = ef[['cropLand', 'grazingLand', 'forestLand', 'fishingGround', 'builtupLand', 'carbon']]
y = ef['OD']

m = model_gradientboosting(X, y, 60, 3, 0.6)

ef_pred = predict_OD(m, df_predictions)
ef_pred['year'] = list(range(1970,2031))
ef_pred.set_index(['year'], inplace=True)

#transform number to date
trans_OD = make_dateofyear(ef_pred['prediction'], ef_pred.index)
ef_pred['OD_pred'] = trans_OD

#save df
ef_pred.to_csv('./data_csv/df_02_full.csv')

#create df only with predictions:
ef_pred_02 = ef_pred.drop(index=list(range(1970,2018)), columns=ef_pred.columns[0:6])
ef_pred_02.rename(columns={'prediction':'sce02_pred', 'OD_pred':'sce02_OD'}, inplace=True)

ef_pred_02.to_csv('../dashboard/load_data/data_db/df_02.csv')



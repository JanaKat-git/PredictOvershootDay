'''
Predict OvershootDay for sceanrio01: trends continue.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model_functions import prepare_df, make_dateofyear, model_gradientboosting, predict_OD, trend_feature

#Create df for prediction (generate data)
df_OD = prepare_df('./../data/deficit/world_ef_bc_tot.csv', 'eu_perpared_pc.csv', 'total')
ef = df_OD.drop(index=[1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969])

#check for correlations:
sns.heatmap(round(ef.corr(), 2), annot=True)
plt.show()
plt.clf()

#generate new DataFrame
df_predictions = pd.DataFrame(index=np.arange(61))

for col in ef.columns[0:6]:
    _, pred = trend_feature(ef, col, 'pred', 2031, 'linear', 0, 'tot', 1)
    col_name = col+'_pred'
    df_predictions[col_name] = pred

#Set Index of df_predictions to years
df_predictions['years'] = range(1970, 2031)
df_predictions.set_index(['years'], inplace=True)

#GrazingLand Trend: manually
df_gL, pred_t = trend_feature(ef, 'grazingLand', 'pred_gL', 2031, 'poly',2, 'tot', 1)
df_gL['pred_gL'][25::] = ef['grazingLand'][25::].mean()
plt.plot(ef.index, ef[['grazingLand']])
plt.plot(df_gL.index, df_gL['pred_gL'])
plt.legend(['grazingLand', 'pred'])
plt.xlabel('years')
plt.ylabel('*10⁹ global hectars')
plt.title('GrazingLand (trend manually)')
plt.savefig('./plots_jpg/world_tot_manual_grzingLand_pred.jpg')
plt.show()

df_predictions['grazingLand_pred'] = df_gL['pred_gL']

#Carbon Trend: polynomial
df_carbon, pred_t = trend_feature(ef, 'carbon', 'pred_carbon', 2031, 'poly',2 , 'tot', 1)
df_predictions['carbon_pred'] = df_carbon['pred_carbon']

#Prediction of the OvershootDay
X = ef[['cropLand', 'grazingLand', 'forestLand', 'fishingGround', 'builtupLand', 'carbon']]
y = ef['OD']

m = model_gradientboosting(X, y, 50, 2, 0.6)

#create new DataFrame with predictes values
ef_pred = predict_OD(m, df_predictions)
ef_pred['year'] = list(range(1970,2031))
ef_pred.set_index(['year'], inplace=True)

#transform calender days in dates
trans_OD = make_dateofyear(ef_pred['prediction'], ef_pred.index)
ef_pred['OD_pred'] = trans_OD

#calculate error between original data from 1971-2017
error = (ef['OD'] - ef_pred['prediction']).mean()

#save df
ef_pred.to_csv('./data_csv/df_01_full.csv')

#create df only with predictions:
ef_pred_01 = ef_pred.drop(index=list(range(1970,2018)), columns=ef_pred.columns[0:6])
ef_pred_01.rename(columns={'prediction':'sce01_pred', 'OD_pred':'sce01_OD'}, inplace=True)
ef_pred_01.to_csv('../dashboard/load_data/data_db/df_01.csv')


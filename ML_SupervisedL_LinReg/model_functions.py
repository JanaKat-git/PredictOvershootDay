'''
Machine Learning Model to predict the OvershootDay
with a Regession model (GradientBoosting) and functions to generate Data for predictions.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import cross_validate
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures

def prepare_df(csv_file_input, csv_file_output, data_input):
    '''
    Prepares a pd.DataFrame for a Regeression model

    Paramteres
    -------
    csv_file_input: str
        path of the input csv-file
    csv_file_output: str
        path of the output csv-file
    data_input: str
        'total' or 'pc' unit of the data (in total or per person)

    Return
    ef: pd.DataFrame
        prepared DataFrame
    -------
    '''
    #load data
    df = pd.read_csv(csv_file_input)
        
    #drop unnecessary columns for the model
    df.drop(columns=['Unnamed: 0', 'id', 'version', 'countryCode', 'countryName',
        'shortName', 'isoa2', 'score'], inplace=True)

    if data_input == 'total':
        bc = df[df['record'] == 'BiocapTotGHA']
        ef = df[df['record'] == 'EFConsTotGHA']

    elif data_input == 'pc':
        bc = df[df['record'] == 'BiocapPerCap']
        ef = df[df['record'] == 'EFConsPerCap']
    
    else:
        ...

    ef.rename(columns={'value': 'ef'}, inplace=True)
    bc.rename(columns={'value': 'bc'}, inplace=True)

    ef['bc'] = list(bc['bc']) #Merge ef and bc in one DataFrame
    ef.set_index('year', inplace=True) 
    ef.drop(columns=['record'], inplace=True) 

    lst = []
    for i in range(0,len(ef)):
        if ef['bc'].iloc[i] >= ef['ef'].iloc[i] :
            lst.append(0)
        else:
            val  = round((ef['bc'].iloc[i]  / ef['ef'].iloc[i]  *365),0)
            lst.append(val)

    ef['OD'] = lst

    lst_dY=[]
    for year, i in zip(ef.index,ef['OD']):
            if i > 0:
                day_str = str(i)[:-2]
                res = datetime.strptime(str(year) + "-" + day_str, "%Y-%j").strftime("%m-%d-%Y")
                lst_dY.append(res)
            else:
                lst_dY.append(0)

    ef['OD_year'] = lst_dY

    ef['OD_year'] = pd.to_datetime(ef['OD_year'])

    ef.drop(columns=['ef','bc'], inplace = True)

    if data_input == 'total':
        for col in ef.columns[0:-2]:
            ef[col] = round((ef[col]/10e+09),4)
        
    else:
        ...
    
    ef.to_csv(csv_file_output)

    return ef


def make_dateofyear(data_lst, year_lst):
    '''
    Takes a list of numbers and a list of years and calculates the date of the year for this year:
    e.g.: number = 365, year= 2000 --> 31.12.2000

    Paramter
    --------
    data_lst: list
        List with numbers representing the day of the year.
    year_lst: list
       List with numbers representing the corresponding years to data_lst.
    
    Return
    -------
    lst_dates:list
        List with transformed dates.
    '''

    lst_dates=[]
    for year, calenderday in zip(year_lst,data_lst):
        if 0 < calenderday < 365: #check if calenderday is within a year, values >= 365 means that there is no OvershootDay for this year
            day_str = str(calenderday)[:-2]
            result = datetime.strptime(str(year) + "-" + day_str, "%Y-%j").strftime("%Y-%m-%d")
            lst_dates.append(result)
        else:
            lst_dates.append(0)
    
    return lst_dates 


def model_gradientboosting(X, y, no_n_estimators, no_max_depth, val_alpha):
    '''
    Build and train a Gradient Boosting Model to predict OD using yearly data of 
    'cropLand', 'grazingLand', 'forestLand', 'fishingGround', 'builtupLand', 'carbon'.

    Paramter
    --------
    X: pd.DataFrame
        DataFrame with X matrix for model.
    y: pd.DataFrame
        DataFrame with y scalar for model.
    no_n_estimators: int
        # of n_estimators for the gradient-boosted trees.
    no_max_depth: int
        # of max_depth for the gradient-boosted trees.
    val_alpha: int  
        aplha value for the gradient-boosted trees.
    
    Return
    -------
    model: sklearn.model
        trained model using to predict OD.
    '''
    #Train Test Split
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=31)

    model = GradientBoostingRegressor(n_estimators=no_n_estimators, max_depth=no_max_depth, alpha=val_alpha)
    model.fit(Xtrain, ytrain)
    
    cv = cross_validate(estimator=model,
                X=Xtrain,
                y=ytrain,
                cv=4,
                return_train_score=True)

    print('GradientBoosting cross validation: TEST '+str(round(cv['test_score'].mean(),2)) +' TRAIN '+ str(round(cv['train_score'].mean(),2)))

    return model


def predict_OD(model, df):
    '''
    Predict OD using data in DataFrame. 

    Paramter
    --------
    model: sklearn.model
        model using to predict OD.
    df: pd.DataFrame
        dataframe with data ('cropLand', 'grazingLand', 'forestLand', 
        'fishingGround', 'builtupLand', 'carbon') for the prediction.
    
    Return
    -------
    df: pd.DataFrame
        dataframe with prdicted values.
    '''
    #make a prediction
    ypred = model.predict(df)

    #append the df with the prediction as number
    df['prediction'] = ypred.round(0)

    return df


def trend_feature(df, feature, feature_pred, year_end, dimension, poly_degree, normal, factor):
    '''
    Calculate the trend for a choosen feature. 

    Paramter
    -------
    df: pd.DataFrame
        DataFrame for the calaculation of the Trend.
    feature: str    
        Name of the Feature to calculate the trend.
    feature_pred: str
        Name of the column for the predicted values in the DataFrame. 
    year_end: int
        Year endpoint of the yearrange for prediction.
    dimension: str
        'linear' or 'poly' for LinearReg
    poly_degree: int
        degree for polynomial feature
    normal: str 
        'tot' for total data or 'pc' for data per person
    factor: int
        factor of the increase or decrease of development (e.g. 1.05 increase by 5 %)

    Return
    -------
    df: pd.DataFrame
        DataFrame with predicted values.
    predictions: list
        predicted values as list.
    '''
    
    #create Timesteps to create Trend
    df['timestep'] = range(len(df))
    Xtrend = df[['timestep']]

    if dimension is 'linear':
        m_lin = LinearRegression()
        m_lin.fit(Xtrend, df[[feature]])

        slope = m_lin.coef_ 
        intercept = m_lin.intercept_
      
        number_new_rows = len(range(1970, year_end)) - len(df)
        if number_new_rows > 0:
            for each_item in range(number_new_rows): 
                df = df.append(pd.Series(), ignore_index=True)
        else:
            ...

        #calculate values
        predictions = []
        for number in range(len(range(1970, year_end))):
            pred = float(slope[0])*factor*number + float(intercept[0])
            predictions.append(pred)

        df[feature_pred] = predictions

    elif dimension is 'poly':
        column_poly = ColumnTransformer([('poly', PolynomialFeatures(degree=poly_degree, include_bias=False), ['timestep'])]) 

        column_poly.fit(Xtrend) 
        Xtrans=column_poly.transform(Xtrend) #transform xdata

        m_poly = LinearRegression()
        m_poly.fit(Xtrans, df[[feature]])

        slope = m_poly.coef_ 
        intercept = m_poly.intercept_
            
        number_new_rows = len(range(1970, year_end)) - len(df)
        if number_new_rows > 0:
            for times in range(number_new_rows): 
                df = df.append(pd.Series(), ignore_index=True)
        else:
            ...

        #calculate values
        predictions = []
        for number in range(len(range(1970, year_end))):
            pred = float(slope[0][0])*factor*(number) + float(slope[0][1])*(number**2) + float(intercept[0])
            predictions.append(pred)

        df[feature_pred] = predictions
    
    df['years'] = range(1970, year_end)
    df.set_index(['years'], inplace=True)
    
    #Plot values
    plt.plot(df.index, df[[feature]])
    plt.plot(df.index, df[feature_pred])
    plt.legend([feature, feature_pred])
    plt.xlabel('years')
    plt.title(feature)
    if normal is 'tot':
        plt.ylabel('*10⁹ global hectares')
        plt.savefig('./plots_jpg/world_tot_'+dimension+'_'+feature+'_pred.jpg')
        plt.show()
    elif normal is 'pc':
        plt.ylabel('global hectars per person')
        plt.savefig('./plots_jpg/world_pc_'+dimension+'_'+feature+'_pred.jpg')
        plt.show()
        
    plt.clf

    return df, predictions


def calculate_increase_percent(columnname):
    '''
    Calcultes the increase of the feature (columnname) in percent over the given years.
    '''
    return ((columnname.iloc[-1] / columnname.iloc[0])-1) * 100


def calculate_scenarios(df, factor_cL, factor_gL, factor_fL, factor_fG, factor_bL, factor_C, year_end, number_rows):
    '''
    Calculate the values for the given sceanrio(using the slope of the trend) and save the data in an DataFrame. 

    Paramter
    -------
    df: pd.DataFrame
        DataFrame of the originall data.
    factor_cL: float    
        Factor for the change of the cropLand values (multiplication of the slope) 
        given as:1.00 + change (1.05 --> increase of 5 %).
    factor_gL: float    
            Factor for the change of the grazingLand values see factor_cL
    factor_fL: float    
            Factor for the change of the forestLand values see factor_cL
    factor_fG: float    
            Factor for the change of the fishingGround values see factor_cL
    factor_bL: float    
            Factor for the change of the builtupLand values see factor_cL
    factor_C: float    
            Factor for the change of the Carbon values see factor_cL
    year_end: int
            Endyear for datarange
    number_rows: int
            Number of new rows in the dataframe(size of datarange for prediction)
    
    Return
    -------
    df_predictions: pd.DataFrame
        DataFrame with predicted values.

    '''
    df_predictions = pd.DataFrame(index=np.arange(number_rows))
    predictions = []
    _, pred_cL = trend_feature(df, 'cropLand', 'pred', year_end, 'linear',0 , 'tot', factor_cL)
    df_gL, pred_gL = trend_feature(df, 'grazingLand', 'pred', year_end, 'linear',0 , 'tot', factor_gL)
    df_gL['pred'].iloc[25::] = df['grazingLand'][25::].mean()
    _, pred_fL = trend_feature(df, 'forestLand', 'pred', year_end, 'linear',0 , 'tot', factor_fL)
    _, pred_fG = trend_feature(df, 'fishingGround', 'pred', year_end, 'linear',0 , 'tot', factor_fG)
    _, pred_bL = trend_feature(df, 'builtupLand', 'pred', year_end, 'linear', 0, 'tot', factor_bL)
    _, pred_c = trend_feature(df, 'carbon', 'pred', year_end, 'poly', 2, 'tot', factor_C)

    predictions.append([pred_cL, pred_gL, pred_fL, pred_fG, pred_bL, pred_c])
    for col, i in zip(df.columns[0:6],range(6)):
        df_predictions[col+'_pred'] = predictions[0][i]
    
    df_predictions['years'] = range(1970, year_end)
    df_predictions.set_index(['years'], inplace=True)
    
    return df_predictions


def calculate_scenario_all(df, factor_all,  year_end, number_rows):
    '''
    Calculate the values for the given sceanrio (using the last value of the dataset as initial value
    and calculate a yearly increase/decrease for the upcoming years) for all features and save the data in an DataFrame. 

    Paramter
    -------
    df: pd.DataFrame
        DataFrame of the originall data.
    factor_all: float    
        Factor for the yearly change of the values (last value of the Dataset as inital value) 
        given as:1.00 +/- change (0.98 --> decrease by 2 %).
    year_end: int
            Endyear for datarange
    number_rows: int
            Number of new rows in the dataframe(size of datarange for prediction)
    
    Return
    -------
    df_predictions: pd.DataFrame
        DataFrame with predicted values.

    '''
    df_predictions = pd.DataFrame(index=np.arange(number_rows), columns=df.columns[0:6])

    df_predictions['cropLand'].iloc[0] = df.iloc[-1,0]
    df_predictions['grazingLand'].iloc[0] = df.iloc[-1,1]
    df_predictions['forestLand'].iloc[0] = df.iloc[-1,2]
    df_predictions['fishingGround'].iloc[0] = df.iloc[-1,3]
    df_predictions['builtupLand'].iloc[0] = df.iloc[-1,4]
    df_predictions['carbon'].iloc[0] = df.iloc[-1,5]

    for column in df_predictions.columns:
        for i in range(1,13):
            df_predictions[column].iloc[i] = round(df_predictions[column].iloc[i-1]*factor_all,4)

    df_predictions['year'] = list(range(2018,year_end))
    df_predictions.set_index(['year'], inplace=True)

    df_predictions = df.append(df_predictions)

    df_predictions.drop(columns=['OD','OD_year'], inplace=True)

    return df_predictions

def calculate_scenario_feature(filepath, lastyear_dataset, factor_feature,  name_feature_pred):
    '''
    Calculate the values for the given sceanrio (using the last value of the dataset as initial value
    and calculate a yearly increase/decrease for the upcoming years) for one feature (all the other features develop with the trend of the last years)
    and save the data in an DataFrame. 

    Paramter
    -------
    filepath: str
        DataFrame of the data with the predicted values when all features continues to develop with the trend. 
    lastyear_dataset: int   
        Last year of the unpredicted values (Last yearly value of the original dataset without predictions)
    factor_featuret: float   
        Factor for the yearly change of the value (last value of the Dataset as inital value) 
        given as:1.00 +/- change (0.98 --> decrease by 2 %).
    name_feature_pred: str
            Columnname of he predicted feature as 'feature_pred'
    
    Return
    -------
    df_prediction: pd.DataFrame
        DataFrame with predicted values for chosen feature and for the other features the trend continiues. 

    '''

    df_prediction = pd.read_csv(filepath) #load data with normal trend continue

    years_pred = df_prediction['year'].iloc[-1] - lastyear_dataset

    for i in range(-years_pred ,0):
        df_prediction[name_feature_pred].iloc[i] = round(df_prediction[name_feature_pred].iloc[i-1]*factor_feature,4)

    df_prediction['year'] = list(range(1970,df_prediction['year'].iloc[-1]+1))
    df_prediction.set_index(['year'], inplace=True)

    df_prediction.drop(columns=['prediction','OD_pred'], inplace=True)

    return df_prediction
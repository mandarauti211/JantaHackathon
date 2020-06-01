# Importing packages

import pandas as pd
import numpy as np
import calendar
from datetime import datetime
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Importing file

mydateparser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
Data = pd.read_csv(r'C:\Users\manda\Downloads\Data Science data sets\analytics Vidya Hackathon\train_aWnotuB.csv',
                   parse_dates = ['DateTime'],
                   date_parser = mydateparser)

print(Data.iloc[2:5,:])

# Creating Month day year time level variables
print(type(Data['DateTime']))

Data['DateTime'] = pd.to_datetime(Data['DateTime'])
Data['year'] = Data['DateTime'].dt.year
Data['month'] = Data['DateTime'].dt.month
Data['day'] = Data['DateTime'].dt.day
Data['weekday'] = Data['DateTime'].dt.dayofweek
Data['hour'] = Data['DateTime'].dt.hour
Data['hour'] = Data['hour'].apply(pd.to_numeric).astype(int)
Data['year_month'] = Data['year'].astype(str) + "_" + Data['month'].astype(str)

print(Data.iloc[2:5,:])



def hour_band(x):
    if (x['hour'] >= 0 and x['hour'] <=6) :
        return 0
    elif (x['hour'] > 6 and x['hour'] <=12):
        return 2
    elif (x['hour'] > 12 and x['hour'] <= 16):
        return 1
    elif (x['hour'] > 16 and x['hour'] <= 23):
        return 3
    
def week_band(x):
    if ( (x['weekday'] == 5) | (x['weekday'] == 6)) :
        return 1
    else: 
        return 0

    
Data = Data.assign(hourband = Data.apply(hour_band,axis=1))
Data = Data.assign(weekband = Data.apply(week_band,axis=1))
print(Data.iloc[2:5,:])

#(Data['year_month'] == '2016_7') |  (Data['year_month'] == '2016_8') | (Data['year_month'] == '2016_9') |  (Data['year_month'] == '2016_10')
##############################Model Fitting Junction1################

###############Junction123
Data_junc123 = Data.loc[(Data['year'] == 2017) & ]
X , y=  Data_junc123.loc[:,['month','day','hour','weekday','Junction','year']] , Data_junc123.loc[:,['Vehicles']]

Data_dmatrix = xgb.DMatrix(data=X,label=y)
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

xg_reg_Junc123 = xgb.XGBRegressor(objective = 'reg:squarederror', colsample_bytree = 1, learning_rate = 0.07, max_depth = 7,
                          alpha = 5, n_estimators = 200)

xg_reg_Junc123.fit(X_train,y_train)
preds = xg_reg_Junc123.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,preds))
print("RMSE : %f" %(rmse))

#
################Junction2
#Data_junc2 = Data.loc[(Data['year'] == 2017) & (Data['Junction'] == 2)]
#X , y=  Data_junc2.loc[:,['month','day','hour','weekday','hourband','weekband']] , Data_junc2.loc[:,['Vehicles']]
#
#Data_dmatrix = xgb.DMatrix(data=X,label=y)
#X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
#
#xg_reg_Junc2 = xgb.XGBRegressor(objective = 'reg:squarederror', colsample_bytree = 1, learning_rate = 0.1, max_depth = 3,
#                          alpha = 10, n_estimators = 1200)
#
#xg_reg_Junc2.fit(X_train,y_train)
#preds = xg_reg_Junc2.predict(X_test)
#
#rmse = np.sqrt(mean_squared_error(y_test,preds))
#print("RMSE : %f" %(rmse))
#
################Junction3
#Data_junc3 = Data.loc[(Data['year'] == 2017) & (Data['Junction'] == 3)]
#X , y=  Data_junc3.loc[:,['month','day','hour','weekday','hourband','weekband']] , Data_junc3.loc[:,['Vehicles']]
#
#Data_dmatrix = xgb.DMatrix(data=X,label=y)
#X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
#
#xg_reg_Junc3 = xgb.XGBRegressor(objective = 'reg:squarederror', colsample_bytree = 1, learning_rate = 0.1, max_depth = 6,
#                          alpha = 10, n_estimators = 1200)
#
#xg_reg_Junc3.fit(X_train,y_train)
#preds = xg_reg_Junc3.predict(X_test)
#
#rmse = np.sqrt(mean_squared_error(y_test,preds))
#print("RMSE : %f" %(rmse))

###############Junction4
Data_junc4 = Data.loc[(Data['year'] == 2017) & (Data['Junction'] == 4)]
X , y=  Data_junc4.loc[:,['month','day','hour','weekday','Junction','year']] , Data_junc4.loc[:,['Vehicles']]

Data_dmatrix = xgb.DMatrix(data=X,label=y)
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

xg_reg_Junc4 = xgb.XGBRegressor(objective = 'reg:squarederror', colsample_bytree = 1, learning_rate = 0.07, max_depth = 7,
                          alpha = 5, n_estimators = 200)

xg_reg_Junc4.fit(X_train,y_train)
preds = xg_reg_Junc4.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,preds))
print("RMSE : %f" %(rmse))


#############Importing test File###########################

Test = pd.read_csv(r'C:\Users\manda\Downloads\Data Science data sets\analytics Vidya Hackathon\test_BdBKkAj_L87Nc3S.csv',
                   parse_dates = ['DateTime'],
                   date_parser = mydateparser)

Test.iloc[1:10,:]

Test['DateTime'] = pd.to_datetime(Test['DateTime'])
Test['year'] = Test['DateTime'].dt.year
Test['month'] = Test['DateTime'].dt.month
Test['weekday'] = Test['DateTime'].dt.dayofweek
Test['day'] = Test['DateTime'].dt.day
Test['hour'] = Test['DateTime'].dt.hour
Test['hour'] = Test['hour'].apply(pd.to_numeric).astype(int)

Test = Test.assign(hourband = Test.apply(hour_band,axis=1))
Test = Test.assign(weekband = Test.apply(week_band,axis=1))
Test.iloc[1:10,:]

X_test_final = Test.loc[:,['month','day','hour','weekday','hourband','weekband','Junction']]

####Scoring Junction123
X_test_final_Junc123 = X_test_final.loc[(X_test_final['Junction'] == 1) | (X_test_final['Junction'] == 2) | (X_test_final['Junction'] == 3)]
X_test_final_Junc123 = X_test_final_Junc123.loc[:,['month','day','hour','weekday','Junction','year']]
y_test_final_Junc123 = xg_reg_Junc123.predict(X_test_final_Junc123)
y_test_final_Junc123 = pd.DataFrame(data = y_test_final_Junc123, columns = ['Vehicles'])

#####Scoring Junction2
#X_test_final_Junc2 = X_test_final.loc[X_test_final['Junction'] == 2]
#X_test_final_Junc2 = X_test_final_Junc2.loc[:,['month','day','hour','weekday','hourband','weekband']]
#y_test_final_Junc2 = xg_reg_Junc2.predict(X_test_final_Junc2)
#y_test_final_Junc2 = pd.DataFrame(data = y_test_final_Junc2, columns = ['Vehicles'])
#
#####Scoring Junction3
#X_test_final_Junc3 = X_test_final.loc[X_test_final['Junction'] == 3]
#X_test_final_Junc3 = X_test_final_Junc3.loc[:,['month','day','hour','weekday','hourband','weekband']]
#y_test_final_Junc3 = xg_reg_Junc3.predict(X_test_final_Junc3)
#y_test_final_Junc3 = pd.DataFrame(data = y_test_final_Junc3, columns = ['Vehicles'])

####Scoring Junction3
X_test_final_Junc4 = X_test_final.loc[X_test_final['Junction'] == 4]
X_test_final_Junc4 = X_test_final_Junc4.loc[:,['month','day','hour','weekday','Junction','year']]
y_test_final_Junc4 = xg_reg_Junc4.predict(X_test_final_Junc4)
y_test_final_Junc4 = pd.DataFrame(data = y_test_final_Junc4, columns = ['Vehicles'])

Data_submission = pd.concat([y_test_final_Junc123, y_test_final_Junc4],axis=0)
Data_submission['Vehicles'] = Data_submission['Vehicles'].astype(int)
Data_submission.reset_index(drop = True, inplace=True)
Test.reset_index(drop = True, inplace=True)

Data_submission2 = pd.concat([Test,Data_submission],axis=1)

Data_submission3 = Data_submission2.loc[:,['ID','Vehicles']]

Data_submission3.to_csv('C:/Users/manda/Downloads/Data Science data sets/analytics Vidya Hackathon/submission45.csv', 
                        index=False,
                        encoding='utf-8')

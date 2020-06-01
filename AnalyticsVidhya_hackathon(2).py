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
from sklearn.preprocessing import LabelEncoder

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


def hour_band(x):
    if (x['hour'] >= 0 and x['hour'] <=6) :
        return 0
    elif (x['hour'] > 6 and x['hour'] <=12):
        return 3
    elif (x['hour'] > 12 and x['hour'] <= 16):
        return 2
    elif (x['hour'] > 16 and x['hour'] <= 21):
        return 3
    elif (x['hour'] > 21 and x['hour'] <= 23):
        return 1
    
def week_band(x):
    if ( (x['weekday'] == 5) | (x['weekday'] == 6)) :
        return 1
    else: 
        return 0


    
Data = Data.assign(hourband = Data.apply(hour_band,axis=1))
Data = Data.assign(weekband = Data.apply(week_band,axis=1))
print(Data.iloc[2:5,:])

Data = Data.loc[(Data['year']==2017)]
# Data For Different Junctions
Data['Vehicles'] = Data['Vehicles'].apply(pd.to_numeric)


#Model Fitting

X , y=  Data.loc[:,['month','day','hourband','weekday','Junction','hour','weekband']] , Data.loc[:,['Vehicles']]

Data_dmatrix = xgb.DMatrix(data=X,label=y)
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

xg_reg = xgb.XGBRegressor(objective = 'reg:squarederror', colsample_bytree = 1, learning_rate = 0.1, max_depth = 6,
                          alpha = 10, n_estimators = 1200)

xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)

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
Data['hour'] = Data['hour'].apply(pd.to_numeric).astype(int)

Test = Test.assign(hourband = Test.apply(hour_band,axis=1))
Test.iloc[1:10,:]

X_test_final = Test.loc[:,['month','day','hourband','weekday','Junction','hour','weekband']]
Y_test_final = xg_reg.predict(X_test_final)
X_test_final.iloc[1:10,:]

Y_test_final_pd = pd.DataFrame(data = Y_test_final, columns = ['Vehicles'])
Y_test_final_pd['Vehicles'] = Y_test_final_pd['Vehicles'].astype(int)

Data_submission = pd.concat([Test,Y_test_final_pd],axis=1)
Data_submission2 = Data_submission.loc[:,['ID','Vehicles']]

Data_submission2.to_csv('C:/Users/manda/Downloads/Data Science data sets/analytics Vidya Hackathon/Submission19.csv', 
                        index=False,
                        encoding='utf-8')

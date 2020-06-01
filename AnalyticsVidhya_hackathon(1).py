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

print(Data.iloc[2:5,:])


# Data For Different Junctions
Data['Vehicles'] = Data['Vehicles'].apply(pd.to_numeric)


#Visualization

fig, ax = plt.subplots(figsize=(2,5))
Data.groupby(['year']).sum()['Vehicles'].plot(ax=ax)

fig, ax = plt.subplots(figsize=(12,5))
Data.groupby(['month']).sum()['Vehicles'].plot(ax=ax)

fig, ax = plt.subplots(figsize=(12,5))
Data.groupby(['day']).sum()['Vehicles'].plot(ax=ax)

fig, ax = plt.subplots(figsize=(12,5))
Data.groupby(['hour']).sum()['Vehicles'].plot(ax=ax)

fig, ax = plt.subplots(figsize=(6,5))
Data.groupby(['weekday']).sum()['Vehicles'].plot(ax=ax)

fig, ax = plt.subplots(figsize=(3,5))
Data.groupby(['Junction']).sum()['Vehicles'].plot(ax=ax)


#Model Fitting

X , y=  Data.loc[:,['year','month','day','hour','weekday','Junction']] , Data.loc[:,['Vehicles']]

Data_dmatrix = xgb.DMatrix(data=X,label=y)
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

xg_reg = xgb.XGBRegressor(objective = 'reg:squarederror', colsample_bytree = 1, learning_rate = 0.05, max_depth = 6,
                          alpha = 10, n_estimators = 1000)

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
Test['hour'] = Test['DateTime'].dt.hour
Test['day'] = Test['DateTime'].dt.day

Test.iloc[1:10,:]


X_test_final = Test.loc[:,['year','month','day','hour','weekday','Junction']]
Y_test_final = xg_reg.predict(X_test_final)
X_test_final.iloc[1:10,:]

Y_test_final_pd = pd.DataFrame(data = Y_test_final, columns = ['Vehicles'])
Y_test_final_pd['Vehicles'] = Y_test_final_pd['Vehicles'].astype(int)

Data_submission = pd.concat([Test,Y_test_final_pd],axis=1)
Data_submission2 = Data_submission.loc[:,['ID','Vehicles']]

Data_submission2.to_csv('C:/Users/manda/Downloads/Data Science data sets/analytics Vidya Hackathon/Submission4.csv', 
                        index=False,
                        encoding='utf-8')
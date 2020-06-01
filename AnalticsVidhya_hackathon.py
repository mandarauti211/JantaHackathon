# Importing packages

import pandas as pd
import numpy as np
import calendar
from datetime import datetime
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Importing file

mydateparser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
Data = pd.read_csv(r'C:\Users\manda\Downloads\Data Science data sets\analytics Vidya Hackathon\train_aWnotuB.csv',
                   index_col = ['ID'],
                   parse_dates = ['DateTime'],
                   date_parser = mydateparser)

print(Data.iloc[2:5,:])


# Creating Month day year time level variables
print(type(Data['DateTime']))

Data['DateTime'] = pd.to_datetime(Data['DateTime'])
Data['year'] = Data['DateTime'].dt.year
Data['month'] = Data['DateTime'].dt.month
Data['day'] = Data['DateTime'].dt.day

Data['hour'] = Data['DateTime'].dt.hour

print(Data.iloc[2:5,:])


# Data For Different Junctions
Data[['Vehicles','year','month','day','hour']] = Data[['Vehicles','year','month','day','hour']].apply(pd.to_numeric)
Data_junc1 = Data.loc[Data['Junction']==1]
print(Data_junc1.iloc[1:10,:])

Data_junc2 = Data.loc[Data['Junction']==2]
print(Data_junc2.iloc[1:10,:])

Data_junc3 = Data.loc[Data['Junction']==3]
print(Data_junc3.iloc[1:10,:])

Data_junc4 = Data.loc[Data['Junction']==4]
print(Data_junc4.iloc[1:10,:])
Data_junc4['year'].unique()
Data_junc4['month'].unique()


#Visualization

fig, ax = plt.subplots(figsize=(2,5))
Data_junc1.groupby(['year']).sum()['Vehicles'].plot(ax=ax)

fig, ax = plt.subplots(figsize=(12,5))
Data_junc1.groupby(['month']).sum()['Vehicles'].plot(ax=ax)

fig, ax = plt.subplots(figsize=(12,5))
Data_junc1.groupby(['day']).sum()['Vehicles'].plot(ax=ax)

fig, ax = plt.subplots(figsize=(12,5))
Data_junc1.groupby(['hour']).sum()['Vehicles'].plot(ax=ax)


fig, ax = plt.subplots(figsize=(2,5))
Data_junc2.groupby(['year']).sum()['Vehicles'].plot(ax=ax)

fig, ax = plt.subplots(figsize=(12,5))
Data_junc2.groupby(['month']).sum()['Vehicles'].plot(ax=ax)

fig, ax = plt.subplots(figsize=(12,5))
Data_junc2.groupby(['day']).sum()['Vehicles'].plot(ax=ax)

fig, ax = plt.subplots(figsize=(12,5))
Data_junc2.groupby(['hour']).sum()['Vehicles'].plot(ax=ax)


fig, ax = plt.subplots(figsize=(2,5))
Data_junc3.groupby(['year']).sum()['Vehicles'].plot(ax=ax)

fig, ax = plt.subplots(figsize=(12,5))
Data_junc3.groupby(['month']).sum()['Vehicles'].plot(ax=ax)

fig, ax = plt.subplots(figsize=(12,5))
Data_junc3.groupby(['day']).sum()['Vehicles'].plot(ax=ax)

fig, ax = plt.subplots(figsize=(12,5))
Data_junc3.groupby(['hour']).sum()['Vehicles'].plot(ax=ax)


fig, ax = plt.subplots(figsize=(1,5))
Data_junc4.groupby(['year']).sum()['Vehicles'].plot(ax=ax)

fig, ax = plt.subplots(figsize=(12,5))
Data_junc4.groupby(['month']).sum()['Vehicles'].plot(ax=ax)

fig, ax = plt.subplots(figsize=(12,5))
Data_junc4.groupby(['day']).sum()['Vehicles'].plot(ax=ax)

fig, ax = plt.subplots(figsize=(12,5))
Data_junc4.groupby(['hour']).sum()['Vehicles'].plot(ax=ax)

#Model Fitting

X_junc1 , y_junc1 =  Data_junc1.loc[:,['year','month','day','hour']] , Data_junc1.loc[:,['Vehicles']]

Data_junc1_dmatrix = xgb.DMatrix(data=X_junc1,label=y_junc1)
X_train_junc1,X_test_junc1,y_train_junc1,y_test_junc1 = train_test_split(X_junc1, y_junc1, test_size = 0.2, random_state = 123)

xg_reg = xgb.XGBRegressor(objective = 'reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 3,
                          alpha = 10, n_estimators = 10)

xg_reg.fit(X_train_junc1,y_train_junc1)
preds_junc1 = xg_reg.predict(X_test_junc1)

rmse_junc1 = np.sqrt(mean_squared_error(y_test_junc1,preds_junc1))
print("RMSE : %f" %(rmse_junc1))




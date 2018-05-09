import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.multioutput import MultiOutputRegressor
import time
import sys


def grant_predictor(onu_id,onu_df,window,predict,features,model,metric):
    index=0 # window start
    index_max = 0 # prediction end

    # list with metrics of each prediction in different observation windows
    metric_list = []
    reg = MultiOutputRegressor(model)#Implement the model

    while index+window < len(onu_df):
        interval=index+window # window final position

        df_tmp = onu_df.iloc[index:interval] # training dataset
        if interval+predict < len(onu_df): # check if prediction doesnt overflow input data
            index_max = interval+predict
        else:
            index_max = len(onu_df)-1

        # check if features evaluated is simple(counter) else counter+timestamp
        if len(features) == 1:
            X_pred = np.array(onu_df[features].iloc[interval:index_max]).reshape(-1,1)
            if len(X_pred) == 0:
                break
            # fitting the model
            reg.fit(np.array( df_tmp[features] ).reshape(-1,1) , df_tmp[['start','end']])
        else:
            X_pred = onu_df[features].iloc[interval:index_max]
            if len(X_pred) == 0:
                break
            # fitting the model
            reg.fit(df_tmp[features] , df_tmp[['start','end']])

        # make prediction
        pred = reg.predict(X_pred)
        # real values to compare with prediction
        Y_true = onu_df[['start','end']].iloc[interval:index_max]
        # metric calculation
        metric_list.append(metric(Y_true, pred,multioutput='uniform_average'))

        # shift past observations window in p positions
        index += predict

    return metric_list

#settings
model = sys.argv[1]
windows = [3,4,5,8,10,15,20]
predicts = [3,4,5,8,10,15,20]
#feature = ['timestamp','counter']
feature = ['counter']
models = {'ols': linear_model.LinearRegression(),'ridge': linear_model.Ridge(alpha=.5),'lasso':linear_model.Lasso(alpha=.1)}
metrics = {'r2': r2_score,'mse': mse}
best_r2 = {'key': "",'r2':float("-inf")}
best_mse = {'key': "",'mse':float("inf")}

# read dataset file
data = pd.read_csv("data-grant_time.csv")
for w in windows:
    for p in predicts:
        d = {'r2':None,'mse':None} #auxiliary dict to create several dataframes
        for metric in metrics:
            result_list = [] # list of results per metric
            # Split the processed dataset by ONU
            for onu in data['ONU_id'].unique():
                # Create a new pandas DataFrame by ONU with only the columns timestamp, counter, start and end
                onu_df = data[ data['ONU_id'] == onu ][ ['timestamp','counter','start','end'] ]
                # call predictor
                result = grant_predictor(onu,onu_df,w,p,feature,models[model],metrics[metric])
                result_list += result
            if metric == 'r2':
                d['r2'] = result_list
            else:
                d['mse'] = result_list
        # Create a pandas DataFrame containing the metric_list for R2 and MSE
        df = pd.DataFrame(d)

        # Print the DataFrame descriptive statistics.
        print('w={},p={}'.format(w,p))
        print df.describe()
        print ""

        # Update the best metrics r2 and mse (int the actual moment)
        if df['r2'].mean() > best_r2['r2']:
    		best_r2['key'] = 'w={},p={}'.format(w,p)
    		best_r2['r2'] = df['r2'].mean()
    	if df['mse'].mean() < best_mse['mse']:
    		best_mse['key'] = 'w={},p={}'.format(w,p)
    		best_mse['mse'] = df['mse'].mean()

# print the best metrics
print "best r2 = {}".format(best_r2)
print "best mse = {}".format(best_mse)

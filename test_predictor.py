import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse


data = pd.read_csv("data-grant_time.csv")

r2 = open("c_test_predictor_r2_score.csv",'w')
r2.write("ONU_id,OLS_start,OLS_end,Ridge_start,Ridge_end,Lasso_start,Lasso_end\n")
MAE = open("c_test_predictor_mean_absolute_error.csv",'w')
MAE.write("ONU_id,OLS_start,OLS_end,Ridge_start,Ridge_end,Lasso_start,Lasso_end\n")
MSE = open("c_test_predictor_mean_squared_error.csv",'w')
MSE.write("ONU_id,OLS_start,OLS_end,Ridge_start,Ridge_end,Lasso_start,Lasso_end\n")


def grant_predictor(onu_id,onu_df,window=20,predict=5):
    index=0
    index_max = 0
    while index+window < len(onu_df):
        interval=index+window

        df_tmp = onu_df.iloc[index:interval]
        if interval+predict < len(onu_df):
            index_max = interval+predict
        else:
            index_max = len(onu_df)-1

        X_pred = np.array(onu_df['counter'].iloc[interval:index_max]).reshape(-1,1)
        #X_pred = np.array(onu_df['timestamp'].iloc[interval:index_max]).reshape(-1,1)
        #X_pred = onu_df[['timestamp','counter']].iloc[interval:index_max]
        if len(X_pred) == 0:
            break
        #predicting start time
        reg1 = linear_model.LinearRegression()
        reg2 = linear_model.Ridge()
        reg3 = linear_model.Lasso()
        reg1.fit(np.array( df_tmp['counter'] ).reshape(-1,1) , df_tmp['start'])
        reg2.fit(np.array( df_tmp['counter'] ).reshape(-1,1) , df_tmp['start'])
        reg3.fit(np.array( df_tmp['counter'] ).reshape(-1,1) , df_tmp['start'])
        # reg1.fit(np.array( df_tmp['timestamp'] ).reshape(-1,1) , df_tmp['start'])
        # reg2.fit(np.array( df_tmp['timestamp'] ).reshape(-1,1) , df_tmp['start'])
        # reg3.fit(np.array( df_tmp['timestamp'] ).reshape(-1,1) , df_tmp['start'])
        # reg1.fit(df_tmp[['timestamp','counter']] , df_tmp['start'])
        # reg2.fit(df_tmp[['timestamp','counter']] , df_tmp['start'])
        # reg3.fit(df_tmp[['timestamp','counter']] , df_tmp['start'])
        OLS_start_pred = reg1.predict(X_pred)
        Ridge_start_pred = reg2.predict(X_pred)
        Lasso_start_pred = reg3.predict(X_pred)
        Y_true = onu_df['start'].iloc[interval:index_max]
        #r2 score srart
        OLS_start_score = r2_score(Y_true, OLS_start_pred)
        Ridge_start_score = r2_score(Y_true, Ridge_start_pred)
        Lasso_start_score = r2_score(Y_true, Lasso_start_pred)
        #mae start
        OLS_start_mae = mae(Y_true, OLS_start_pred)
        Ridge_start_mae = mae(Y_true, Ridge_start_pred)
        Lasso_start_mae = mae(Y_true, Lasso_start_pred)
        #mse start
        OLS_start_mse = mse(Y_true, OLS_start_pred)
        Ridge_start_mse = mse(Y_true, Ridge_start_pred)
        Lasso_start_mse = mse(Y_true, Lasso_start_pred)


        #predicting end time
        reg1 = linear_model.LinearRegression()
        reg2 = linear_model.Ridge()
        reg3 = linear_model.Lasso()
        reg1.fit(np.array( df_tmp['counter'] ).reshape(-1,1) , df_tmp['end'])
        reg2.fit(np.array( df_tmp['counter'] ).reshape(-1,1) , df_tmp['end'])
        reg3.fit(np.array( df_tmp['counter'] ).reshape(-1,1) , df_tmp['end'])
        # reg1.fit(np.array( df_tmp['timestamp'] ).reshape(-1,1) , df_tmp['end'])
        # reg2.fit(np.array( df_tmp['timestamp'] ).reshape(-1,1) , df_tmp['end'])
        # reg3.fit(np.array( df_tmp['timestamp'] ).reshape(-1,1) , df_tmp['end'])
        # reg1.fit(df_tmp[['timestamp','counter']] , df_tmp['end'])
        # reg2.fit(df_tmp[['timestamp','counter']] , df_tmp['end'])
        # reg3.fit(df_tmp[['timestamp','counter']] , df_tmp['start'])
        OLS_end_pred = reg1.predict(X_pred)
        Ridge_end_pred = reg2.predict(X_pred)
        Lasso_end_pred = reg3.predict(X_pred)
        Y_true = onu_df['end'].iloc[interval:index_max]
        #r2 score end
        OLS_end_score = r2_score(Y_true, OLS_end_pred)
        Ridge_end_score = r2_score(Y_true, Ridge_end_pred)
        Lasso_end_score = r2_score(Y_true, Lasso_end_pred)
        #mae end
        OLS_end_mae = mae(Y_true, OLS_end_pred)
        Ridge_end_mae = mae(Y_true, Ridge_end_pred)
        Lasso_end_mae = mae(Y_true, Lasso_end_pred)
        #mse end
        OLS_end_mse = mse(Y_true, OLS_end_pred)
        Ridge_end_mse = mse(Y_true, Ridge_end_pred)
        Lasso_end_mse = mse(Y_true, Lasso_end_pred)

        index += predict
        r2.write("{},{},{},{},{},{},{}\n".format(onu_id,OLS_start_score,OLS_end_score,Ridge_start_score,Ridge_end_score,Lasso_start_score,Lasso_end_score))
        MAE.write("{},{},{},{},{},{},{}\n".format(onu_id,OLS_start_mae,OLS_end_mae,Ridge_start_mae,Ridge_end_mae,Lasso_start_mae,Lasso_end_mae))
        MSE.write("{},{},{},{},{},{},{}\n".format(onu_id,OLS_start_mse,OLS_end_mse,Ridge_start_mse,Ridge_end_mse,Lasso_start_mse,Lasso_end_mse))


for onu in data['ONU_id'].unique():
    onu_df = data[ data['ONU_id'] == onu ][ ['timestamp','counter','start','end'] ]
    grant_predictor(onu,onu_df)

r2.close()
MAE.close()
MSE.close()

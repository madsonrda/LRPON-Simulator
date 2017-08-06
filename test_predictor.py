import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse


def grant_predictor(onu_id,onu_df,window,predict,features,model,target,metric):
    index=0
    index_max = 0

    metric_list = []
    while index+window < len(onu_df):
        interval=index+window

        df_tmp = onu_df.iloc[index:interval]
        if interval+predict < len(onu_df):
            index_max = interval+predict
        else:
            index_max = len(onu_df)-1

        reg = model
        if len(features) == 1:
            X_pred = np.array(onu_df[features].iloc[interval:index_max]).reshape(-1,1)
            if len(X_pred):
                break
            reg.fit(np.array( df_tmp[features] ).reshape(-1,1) , df_tmp[target])

        else:
            X_pred = onu_df[features].iloc[interval:index_max]
            if len(X_pred):
                break
            reg.fit(df_tmp[features] , df_tmp[target])

        pred = reg.predict(X_pred)
        Y_true = onu_df[target].iloc[interval:index_max]

        metric_list(metric(Y_true, OLS_start_pred))

        index += predict

    return np.mean(metric_list)


exponents = [1160, 2320, 3480]
seeds = [20]
windows = [10,20,30]
predicts = [5,10,15,20]
features = [['counter'],['timestamp','counter']]
models = {'ols': linear_model.LinearRegression(),'ridge': linear_model.Ridge(alpha=.5),'lasso':linear_model.Lasso(alpha=.1)}
metrics = {'r2': r2_score,'mse': mse}
targets = ['start','end']


table = {}

for e in exponents:
    table[e] = {}
    for f in features:
        table[e]["{}".format(f)] = {}
        for t in targets:
            table[e]["{}".format(f)][t]= {'r2': {'name':'','max':float("-inf") }, 'mse': {'name':'','min':float("inf") }}


for exp in exponents:
    for feature in features:
        for model in models:
            for w in windows:
                for p in predicts:
                    for target in targets:
                        for metric in metrics:
                            result_list = []
                            for seed in seeds:
                                data = pd.read_csv("csv/delay/ipact-3-27000-0-100-{}-{}-delay.csv".format(seed, exp))
                                for onu in data['ONU_id'].unique():
                                    onu_df = data[ data['ONU_id'] == onu ][ ['timestamp','counter','start','end'] ]
                                    result = grant_predictor(onu,onu_df,w,p,feature,models[model],target,metrics[metric])
                                    result_list.append(result)
                            mean = np.mean(result_list)
                            name = "{}:w={},p={}".format(model,w,p)
                            if metric == 'r2':
                                if mean > table[exp]["{}".format(feature)][target]['r2']['max']:
                                    table[exp]["{}".format(feature)][target]['r2']['max'] = mean
                                    table[exp]["{}".format(feature)][target]['r2']['name'] = name
                            else:
                                if mean < table[exp]["{}".format(feature)][target]['mse']['min']:
                                    table[exp]['r2']["{}".format(feature)][target]['mse']['min'] = mean
                                    table[exp]['r2']["{}".format(feature)][target]['mse']['name'] = name


print table

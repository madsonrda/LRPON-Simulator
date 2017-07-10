import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

predction_file = open("grant.pred","w")

l= []

data = pd.read_csv("grant_time.csv")

def overlap(l):
    o=0
    j = 1
    for interval in l[:-1]:
        for i in l[j:]:
            if interval[1] > i[0]:
                o+=1
                print("overlap:{}-{}".format(interval,i))
            else:
                break
        j+=1
        # if j== len(l)-1:
        #     break
    print("n overlap: {}".format(o))


def grant_predictor(onu_id,onu_df,window=20,predict=5):
    index=0
    index_max = 0
    while index+window < len(onu_df):
        interval=index+window
        #predicting start time
        df_tmp = pd.DataFrame(onu_df['start'].iloc[index:interval].values, index=list(range(index,interval)), columns=['start'])
        reg = sm.OLS(df_tmp['start'],df_tmp.index).fit()
        if interval+predict < len(onu_df):
            index_max = interval+predict
        else:
            index_max = len(onu_df)-1
        start_pred = reg.predict(range(interval,index_max))

        #predicting end time
        df_tmp = pd.DataFrame(onu_df['end'].iloc[index:interval].values, index=list(range(index,interval)), columns=['end'])
        reg = sm.OLS(df_tmp['end'],df_tmp.index).fit()
        if interval+predict < len(onu_df):
            index_max = interval+predict
        else:
            index_max = len(onu_df)-1
        end_pred = reg.predict(range(interval,index_max))

        #writing
        if len(start_pred) == len(end_pred):
            for i in range(len(start_pred)):
                predction_file.write("{},{},{}\n".format(onu_id,start_pred[i],end_pred[i]))
                l.append( [ start_pred[i], end_pred[i] ] )
        else:
            print("ERROR")


        index += predict


for onu in data['ONU_id'].unique():
    onu_df = data[ data['ONU_id'] == onu ][ ['start','end'] ]
    grant_predictor(onu,onu_df)

l.sort()
overlap(l)

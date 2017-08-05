import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import linear_model

prediction_file = open("grant.pred","w")

l= []
over = []

data = pd.read_csv("data-grant_time.csv")

def overlap(l):
    o=0
    j = 1
    for interval in l[:-1]:
        for i in l[j:]:
            if interval[1] > i[0]:
                o+=1
                #print("overlap:{}-{}".format(interval,i))
                over.append(interval[1] - i[0])
            else:
                break
        j+=1

    print("n overlap: {}".format(o))

def remove_overlap(l):
    j = 1
    for interval1 in l[:-1]:
        for interval2 in l[j:]:
            if interval1[1] > interval2[0]:
                if interval1 in l:
                    index1 = l.index(interval1)
                    new_interval = [ interval1[0] , interval2[0] - 0.000001 ]
                    l[index1] = new_interval

            else:
                break
        j+=1
    return l


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
                prediction_file.write("{},{},{}\n".format(onu_id,start_pred[i],end_pred[i]))
                l.append( [ start_pred[i], end_pred[i] ] )
        else:
            print("ERROR")


        index += predict


for onu in data['ONU_id'].unique():
    onu_df = data[ data['ONU_id'] == onu ][ ['start','end'] ]
    grant_predictor(onu,onu_df)
prediction_file.close()
l.sort()
# l = remove_overlap(l)
overlap(l)
# print set([x for x in l if l.count(x) > 1])
#X_train, X_test, y_train, y_test = train_test_split(np.array(df_tmp.index[30:60]).reshape(-1,1), df_tmp['start'].iloc[30:60], train_size=0.8, test_size=0.2, random_state=42)
serie = pd.DataFrame({"overlap":over})
print serie.describe()

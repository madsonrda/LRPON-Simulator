import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, errno
import sys

try:
    os.makedirs('img')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


IPACT = {}
PD_DBA = {}


load = [25,31,37,43,50,56,62,68,75,81,87,93]
exponents = [1160, 1450, 1740, 2030, 2320, 2610, 2900, 3190, 3480, 3770, 4060, 4350]
seeds = [20]
#windows = [10,20,30]
windows = [30]
predicts = [5,10,15,20]
#predicts = [10]

for w in windows:
    for p in predicts:
        PD_DBA['{}-{}'.format(w,p)] = {}

for exp in exponents:

    ipact = []
    for seed in seeds:
        df_tmp = pd.read_csv("csv/delay/ipact-3-27000-0-100-{}-{}-delay.csv".format(seed, exp))
        ipact.append(df_tmp['delay'].mean()*1000)
    IPACT[exp] = [np.mean(ipact),np.std(ipact)]


for w in windows:
    for p in predicts:
        for exp in exponents:
            pd_dba = []
            for seed in seeds:
                df_tmp = pd.read_csv("csv/delay/pd_dba-3-27000-0-100-{}-{}-{}-{}-delay.csv".format(seed,exp,w,p))
                pd_dba.append(df_tmp['delay'].mean()*1000)
            PD_DBA['{}-{}'.format(w,p)][exp] = [np.mean(pd_dba),np.std(pd_dba)]

ipact_df = pd.DataFrame(IPACT)
pd_dba_df = pd.DataFrame(PD_DBA)

# a = np.array([ i for i in pd_dba_df['10-10'].iloc[:] ])
# print a[:,0]

plt.figure()

title = sys.argv[1]
plt.title(title)
plt.xlabel("load (%)")
plt.ylabel("delay (ms)")

plt.fill_between(load, ipact_df.iloc[0] - ipact_df.iloc[1],ipact_df.iloc[0] + ipact_df.iloc[1], alpha=0.1,color="r")
plt.plot(load, ipact_df.iloc[0], 'o-', color="k",label="IPACT")

for w in windows:
    for p in predicts:
        array = np.array([ i for i in pd_dba_df['{}-{}'.format(w,p)].iloc[:] ])
        plt.fill_between(load, array[:,0] - array[:,1],array[:,0] + array[:,1], alpha=0.1)
        plt.plot(load, array[:,0], '->', label="PD-DBA w={} p={}".format(w,p))

plt.legend(loc='upper center', shadow=True)
plt.savefig("img/Delay-"+title)

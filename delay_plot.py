import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, errno
import sys

#create img directory
try:
    os.makedirs('img')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

#settings
IPACT = {}
PD_DBA = {}
#load % values which represents each exponent
load = [25,31,37,43,50,56,62,68,75,81,87,93]
#pkt arrival distribution exponents
exponents = [1160, 1450, 1740, 2030, 2320, 2610, 2900, 3190, 3480, 3770, 4060, 4350]
# To do a complete test: uncomment the line below
#seeds = [20,30,40,50,60,70,80,90,100,110] #random seeds
# To do a FAST test: uncomment the line below
seeds = [20]
parameters = [{'w':10,'p':5}]# combinations of the parameters w and p simulated
#parameters = [{'w':10,'p':5},{'w':20,'p':20}]# combinations of the parameters w and p simulated

#dictionary to store the delay mean and std by combination of w and p
for param in parameters:
    PD_DBA['{}-{}'.format(param['w'],param['p'])] = {}

#read the ipact delay file for each simulated scenario with different seeds,
# and calculate the mean and std of each seed simulations
for exp in exponents:
    ipact = []
    for seed in seeds:
        df_tmp = pd.read_csv("csv/delay/ipact-3-27000-0-100-{}-{}-delay.csv".format(seed, exp))
        ipact.append(df_tmp['delay'].mean()*1000)
    IPACT[exp] = [np.mean(ipact),np.std(ipact)]

#read the pd_dba delay file for each simulated scenario with different seeds,
# and calculate the mean and std of each seed simulations
for param in parameters :
    for exp in exponents:
        pd_dba = []
        for seed in seeds:
            df_tmp = pd.read_csv("csv/delay/pd_dba-3-27000-0-100-{}-{}-{}-{}-delay.csv".format(seed,exp,param['w'],param['p']))
            pd_dba.append(df_tmp['delay'].mean()*1000)
        PD_DBA['{}-{}'.format(param['w'],param['p'])][exp] = [np.mean(pd_dba),np.std(pd_dba)]

#create a data frame
ipact_df = pd.DataFrame(IPACT)
pd_dba_df = pd.DataFrame(PD_DBA)

#creating figure
plt.figure()

title = sys.argv[1]#figure title from argument
plt.title(title)
plt.xlabel("load (%)")
plt.ylabel("delay (ms)")


plt.errorbar(load, ipact_df.iloc[0].values,ipact_df.iloc[1].values,color="k", linestyle='None')
plt.plot(load, ipact_df.iloc[0], 'o-', color="k",label="IPACT")


number = 4
cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0.25, 1, number)]


for j, param in enumerate(parameters):
    array = np.array([ i for i in pd_dba_df['{}-{}'.format(param['w'],param['p'])].iloc[:] ])

    plt.errorbar(load, array[:,0],array[:,1], color=colors[j],linestyle='None')
    plt.plot(load, array[:,0], '->',color=colors[j] ,label="PD-DBA w={} p={}".format(param['w'],param['p']))

plt.legend(loc='upper center', shadow=True)
plt.savefig("img/Delay-"+title)

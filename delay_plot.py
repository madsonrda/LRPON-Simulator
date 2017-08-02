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

ipact = []
pd_dba = []

load = [25,31,37,43,50,56,62,68,75,81,87,93]
exponents = [1160, 1450, 1740, 2030, 2320, 2610, 2900, 3190, 3480, 3770, 4060, 4350]

for exp in exponents:
    df_tmp = pd.read_csv("csv/delay/ipact-3-27000-0-100-{}-delay.csv".format(exp))
    ipact.append(df_tmp['delay'].mean()*1000)
    df_tmp = pd.read_csv("csv/delay/pd_dba-3-27000-0-100-{}-delay.csv".format(exp))
    pd_dba.append(df_tmp['delay'].mean()*1000)

plt.figure()

title = sys.argv[1]
plt.title(title)
plt.xlabel("load (%)")
plt.ylabel("delay (ms)")

# plt.fill_between(IPACT_LOAD, IPACT_DELAY - IPACT_delay_std,IPACT_DELAY + IPACT_delay_std, alpha=0.1,color="r")
#
# plt.fill_between(PERF_LOAD, PERF_DELAY - PERF_delay_std,PERF_DELAY + PERF_delay_std, alpha=0.1,color="b")
# plt.fill_between(PROP_LOAD, PROP_DELAY - PROP_delay_std_R,PROP_DELAY + PROP_delay_std_R, alpha=0.1,color="g")
plt.plot(load, ipact, 'o-', color="r",label="IPACT")
plt.plot(load, pd_dba, '>-', color="b",label="PD_DBA")
plt.legend(loc='upper center', shadow=True)
plt.savefig("img/Delay-"+title)

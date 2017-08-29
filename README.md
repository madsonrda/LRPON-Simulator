# Long Reach Passive Optical Simulator

This simulator run different Dynamic Bandwidth Allocation (DBA) algorithm to evaluate the average packet delay.

## Getting Started
These instructions will guide you through the steps test the simulator.

### Files

`delay_plot.py`: It saves into the img directory the (delay vs load) line chart of the DBA algorithms.

`ipact.sh`: It runs several simulations of the IPACT algorithm.

`g-sim.py`: The LR-PON simulator.

`pddba.sh`: It runs several simulations of the PD-DBA algorithm.


### Installing

pip install simpy, scipy, sklearn, pandas, matplotlib, statsmodels

### Run the simulator

First simulate the IPACT algorithm:

```
$ bash ipact.sh
```

Then, simulate the PD-DBA algorithm:

```
$ bash pddba.sh
```

Run the following command to plot the delay chart into directory img:

```
python delay_plot.py "LR-PON with 3 ONUs"
```

```
$ ls img
Delay-LR-PON with 3 ONUs.png
```

*Warning*: To simulate using 10 different random seeds you should edit the files ipact.sh, pddba.sh and delay_plot.py, uncommenting the indicated lines. The complete simulations can take more than 1 hour.

#!/bin/bash


for exp in 1160 1740 2320 2900 3480 4060 4640
do
   python g-sim.py pd_dba -o 3 -b 9000 -e $exp &
   python g-sim.py ipact -o 3 -b 9000 -e $exp &
done

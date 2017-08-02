#!/bin/bash


for exp in 1160 1450 1740 2030 2320 2610 2900 3190 3480 3770 4060 4350
do
   python g-sim.py pd_dba -O 3 -b 27000 -e $exp &
done

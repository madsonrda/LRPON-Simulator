#!/bin/bash
# To do a complete test: uncomment the line below
#for seed in 20 30 40 50 60 70 80 90 100 110
# To do a FAST test: uncomment the line below
for seed in 20
do
   for exp in 1160 1450 1740 2030 2320 2610
   do
      python g-sim.py ipact -O 3 -s $seed -e $exp &
   done
   sleep 60

   for exp in 2900 3190 3480 3770 4060 4350
   do
      python g-sim.py ipact -O 3 -s $seed -e $exp &
   done
   sleep 60
done

#!/bin/bash
# To do a complete test: uncomment the line below
for seed in 20 30 40 50 60 70 80 90 100 110
# To do a FAST test: uncomment the line below
#for seed in 20
do
   for pkt in 768000 1536000 3072000 3840000
   do
      python g-sim.py pdba -O 3 -b 9000 -t 5 -T cbr -w 5 -p 5 -s $seed -e $pkt &
   done
   sleep 60
done

#!/bin/bash
# To do a complete test: uncomment the line below
for seed in 20 30 40 50 60 70 80 90 100 110
# To do a FAST test: uncomment the line below
#for seed in 20
do
   for pkt in 768000 1536000 3072000 3840000
   do
      python g-sim.py mdba -O 3 -b $pkt -d 20 -t 5 -T cbr -s $seed -P $pkt &
   done
   sleep 220
done

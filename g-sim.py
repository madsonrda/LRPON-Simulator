import simpy
import random
import functools
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
import logging
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
from sklearn.multioutput import MultiOutputRegressor
import os, errno

#Parsing the inputs arguments
parser = argparse.ArgumentParser(description="Long Reach PON Simulator")
group = parser.add_mutually_exclusive_group()
group.add_argument("-q", "--quiet", action="store_true")
parser.add_argument("A", type=str, default='ipact',choices=["ipact","pd_dba","mdba","mpd_dba"], help="DBA algorithm")
parser.add_argument("-O", "--onu", type=int, default=3, help="The number of ONUs")
parser.add_argument("-b", "--bucket", type=int, default=27000, help="The size of the ONU sender bucket in bytes")
parser.add_argument("-Q", "--qlimit", type=int, default=None ,help="The size of the ONU port queue in bytes")
parser.add_argument("-m", "--maxgrant", type=float, default=0, help="The maximum size of buffer which a grant can allow")
parser.add_argument("-d","--distance", type=int, default=100, nargs='?', help="Distance in km from ONU to OLT")
parser.add_argument("-P","--packetsize", type=int, default=768000, nargs='?', help="Fixed packet size")
parser.add_argument("-e","--exponent", type=float, default=2320, nargs='?', help="Packet arrivals distribution exponent")
parser.add_argument("-s","--seed", type=int, default=20, help="Random seed")
parser.add_argument("-w","--window", type=int, default=10, help="PD-DBA window")
parser.add_argument("-p","--predict", type=int, default=5, help="PD-DBA predictions")
parser.add_argument("-M","--model", type=str, default='ols', choices=["ols","ridge"] ,help="PD-DBA prediction model")
parser.add_argument("-T","--traffic", type=str, default='poisson', choices=["poisson","cbr"] ,help="Traffic distribution")
parser.add_argument("-o", "--output", type=str, default=None, help="Output file name")
parser.add_argument("-t", "--time", type=int, default=30, help="The simulation duration in seconds")
args = parser.parse_args()

#Arguments
DBA_ALGORITHM = args.A
NUMBER_OF_ONUs= args.onu
DISTANCE = args.distance
MAX_GRANT_SIZE = args.maxgrant
MAX_BUCKET_SIZE = args.bucket
ONU_QUEUE_LIMIT = args.qlimit
EXPONENT = args.exponent
FILENAME = args.output
RANDOM_SEED = args.seed
WINDOW = args.window
PREDICT = args.predict
MODEL = args.model
TRAFFIC = args.traffic
SIM_DURATION = args.time
PKT_SIZE = args.packetsize



#settings

MAC_TABLE = {}
Grant_ONU_counter = {}
NUMBER_OF_OLTs = 1

#create directories
try:
    os.makedirs('csv/delay')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs('csv/grant_time')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
try:
    os.makedirs("csv/pkt")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
try:
    os.makedirs("csv/overlap")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

#logging
logging.basicConfig(filename='g-sim.log',level=logging.DEBUG,format='%(asctime)s %(message)s')
if FILENAME:
    delay_file = open("{}-delay.csv".format(FILENAME),"w")
    delay_prediction_file = open("{}-delay_pred.csv".format(FILENAME),"w")
    delay_normal_file = open("{}-delay_normal.csv".format(FILENAME),"w")
    grant_time_file = open("{}-grant_time.csv".format(FILENAME),"w")
    pkt_file = open("{}-pkt.csv".format(FILENAME),"w")
    overlap_file = open("{}-overlap.csv".format(FILENAME),"w")
elif DBA_ALGORITHM == "pd_dba":
    delay_file = open("csv/delay/{}-{}-{}-{}-{}-{}-{}-{}-{}-delay.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT,WINDOW,PREDICT),"w")
    delay_normal_file = open("csv/delay/{}-{}-{}-{}-{}-{}-{}-{}-{}-delay_normal.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT,WINDOW,PREDICT),"w")
    delay_prediction_file = open("csv/delay/{}-{}-{}-{}-{}-{}-{}-{}-{}-delay_pred.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT,WINDOW,PREDICT),"w")
    grant_time_file = open("csv/grant_time/{}-{}-{}-{}-{}-{}-{}-{}-{}-grant_time.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT,WINDOW,PREDICT),"w")
    pkt_file = open("csv/pkt/{}-{}-{}-{}-{}-{}-{}-{}-{}-pkt.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT,WINDOW,PREDICT),"w")
    overlap_file = open("csv/overlap/{}-{}-{}-{}-{}-{}-{}-{}-{}-overlap.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT,WINDOW,PREDICT),"w")
elif DBA_ALGORITHM == "mpd_dba":
    delay_file = open("csv/delay/{}-{}-{}-{}-{}-{}-{}-{}-{}-delay.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT,WINDOW,PREDICT),"w")
    delay_normal_file = open("csv/delay/{}-{}-{}-{}-{}-{}-{}-{}-{}-delay_normal.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT,WINDOW,PREDICT),"w")
    delay_prediction_file = open("csv/delay/{}-{}-{}-{}-{}-{}-{}-{}-{}-delay_pred.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT,WINDOW,PREDICT),"w")
    grant_time_file = open("csv/grant_time/{}-{}-{}-{}-{}-{}-{}-{}-{}-grant_time.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT,WINDOW,PREDICT),"w")
    pkt_file = open("csv/pkt/{}-{}-{}-{}-{}-{}-{}-{}-{}-pkt.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT,WINDOW,PREDICT),"w")
    overlap_file = open("csv/overlap/{}-{}-{}-{}-{}-{}-{}-{}-{}-overlap.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT,WINDOW,PREDICT),"w")
elif DBA_ALGORITHM == "mdba":
    delay_file = open("csv/delay/{}-{}-{}-{}-{}-{}-{}-delay.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT),"w")
    delay_normal_file = open("csv/delay/{}-{}-{}-{}-{}-{}-{}-delay_normal.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT),"w")
    delay_prediction_file = open("csv/delay/{}-{}-{}-{}-{}-{}-{}-delay_pred.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT),"w")
    grant_time_file = open("csv/grant_time/{}-{}-{}-{}-{}-{}-{}-grant_time.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT),"w")
    pkt_file = open("csv/pkt/{}-{}-{}-{}-{}-{}-{}-pkt.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT),"w")
    overlap_file = open("csv/overlap/{}-{}-{}-{}-{}-{}-{}-overlap.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT),"w")
else:
    delay_file = open("csv/delay/{}-{}-{}-{}-{}-{}-{}-delay.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT),"w")
    delay_normal_file = open("csv/delay/{}-{}-{}-{}-{}-{}-{}-delay_normal.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT),"w")
    delay_prediction_file = open("csv/delay/{}-{}-{}-{}-{}-{}-{}-delay_pred.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT),"w")
    grant_time_file = open("csv/grant_time/{}-{}-{}-{}-{}-{}-{}-grant_time.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT),"w")
    pkt_file = open("csv/pkt/{}-{}-{}-{}-{}-{}-{}-pkt.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT),"w")
    overlap_file = open("csv/overlap/{}-{}-{}-{}-{}-{}-{}-overlap.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT),"w")

delay_file.write("ONU_id,delay\n")
delay_normal_file.write("ONU_id,delay\n")
delay_prediction_file.write("ONU_id,delay\n")
grant_time_file.write("source address,destination address,opcode,timestamp,counter,ONU_id,start,end\n")
pkt_file.write("timestamp,adist,size\n")
overlap_file.write("interval\n")

mse_file = open("csv/{}-{}-{}-{}-{}-{}-{}-mse.csv".format(DBA_ALGORITHM,NUMBER_OF_ONUs,MAX_BUCKET_SIZE,MAX_GRANT_SIZE,DISTANCE,RANDOM_SEED,EXPONENT),"w")
mse_file.write("mse_start,mse_end,delay\n")

class ODN(object):
    """This class represents optical distribution Network."""
    def __init__(self, env):
        self.env = env
        self.upstream = []# upstream chanel
        self.downstream = [] # downstream chanel
        #create downstream splitter
        for i in range(NUMBER_OF_ONUs):
            self.downstream.append(simpy.Store(env))
        for i in range(NUMBER_OF_OLTs):
            self.upstream.append(simpy.Store(env))

    def up_latency(self, value,ONU):
        """Calculates upstream propagation delay."""
        yield self.env.timeout(ONU.delay)
        self.upstream[ONU.lamb].put(value)

    def directly_upstream(self,ONU,value):
        self.upstream[ONU.lamb].put(value)

    def down_latency(self,ONU,value):
        """Calculates downstream propagation delay."""
        yield self.env.timeout(ONU.delay)
        self.downstream[ONU.oid].put(value)

    def put_request(self, value,ONU):
        """ONU Puts the Request message in the upstream """
        self.env.process(self.up_latency(value,ONU))

    def get_request(self,lamb):
        """OLT gets the Request message from upstream  """
        return self.upstream[lamb].get()

    def put_grant(self,ONU,value):
        """OLT Puts the Grant message in the downstream """
        self.env.process(self.down_latency(ONU,value))

    def get_grant(self,ONU_id):
        """ONU gets the Grant message from downstream """
        return self.downstream[ONU_id].get()


class Packet(object):
    """ This class represents a network packet """

    def __init__(self, time, size, id, src="a", dst="z"):
        self.time = time# creation time
        self.size = size # packet size
        self.id = id # packet id
        self.src = src #packet source address
        self.dst = dst #packet destination address

    def __repr__(self):
        return "id: {}, src: {}, time: {}, size: {}".\
            format(self.id, self.src, self.time, self.size)

class PacketGenerator(object):
    """This class represents the packet generation process """
    def __init__(self, env, id, ONU, fix_pkt_size=1500, finish=float("inf")):
        self.id = id # packet id
        self.ONU = ONU
        self.env = env # Simpy Environment
        self.fix_pkt_size = fix_pkt_size # Fixed packet size
        self.finish = finish # packe end time
        self.out = None # packet generator output
        self.packets_sent = 0 # packet counter
        self.action = env.process(self.run())  # starts the run() method as a SimPy process


class CBR_PG(PacketGenerator):
    """This class represents the Constant Bit Rate packet generation process """
    def __init__(self,env, id, ONU, fix_pkt_size,interval=0.01):
        self.interval = interval
        PacketGenerator.__init__(self,env, id, ONU, fix_pkt_size)
    def run(self):
        """The generator function used in simulations.
        """
        while self.env.now < self.finish:
            # wait for next transmission
            yield self.env.timeout(self.interval)
            self.packets_sent += 1
            if self.fix_pkt_size:
                p = Packet(self.env.now, self.fix_pkt_size, self.packets_sent, src=self.id)
                pkt_file.write("{},{},{}\n".format(self.env.now,self.interval,self.fix_pkt_size))
            if DBA_ALGORITHM == "mpd_dba" or DBA_ALGORITHM == "mdba":
                msg = {'buffer_size':p.size,'ONU':self.ONU}
                self.ONU.odn.directly_upstream(self.ONU,msg)
            self.out.put(p) # put the packet in ONU port

class poisson_PG(PacketGenerator):
    """This class represents the poisson distribution packet generation process """
    def __init__(self,env, id, ONU, adist, sdist, fix_pkt_size):
        self.arrivals_dist = adist #packet arrivals distribution
        self.size_dist = sdist #packet size distribution
        PacketGenerator.__init__(self,env, id, ONU, fix_pkt_size, finish=float("inf"))
    def run(self):
        """The generator function used in simulations.
        """
        while self.env.now < self.finish:
            # wait for next transmission
            arrival = self.arrivals_dist()
            yield self.env.timeout(arrival)
            self.packets_sent += 1


            if self.fix_pkt_size:
                p = Packet(self.env.now, self.fix_pkt_size, self.packets_sent, src=self.id)
                pkt_file.write("{},{},{}\n".format(self.env.now,arrival,self.fix_pkt_size))
            else:
                size = self.size_dist()
                p = Packet(self.env.now, size, self.packets_sent, src=self.id)
                pkt_file.write("{},{},{}\n".format(self.env.now,arrival,size))
            self.out.put(p) # put the packet in ONU port

# class SubStream(object):
#     """This class represents the sub-streams which will be aggregated by the SelfSimilar class"""
#     def __init__(self,env, on_dist, off_dist,aggregator,size):
#         self.on = on_dist #packet arrivals ON distribution
#         self.off = off_dist #packet arrivals ON distribution
#         self.aggregator = aggregator
#         self.size = size
#         self.packets_sent = 0 # packet counter
#         self.action = env.process(self.run())  # starts the run() method as a SimPy process
#
#     def run(self):
#         while True:
#             on_period = self.env.now + (self.on()/1000)
#             while self.env.now <= on_period:
#                 self.packets_sent += 1
#                 p = Packet(self.env.now, self.size, self.packets_sent, src=self.id)
#                 pkt_file.write("{},{},{}\n".format(self.env.now,on_period,size))
#                 bits = p.size * 8
#                 sending_time = 	bits/float(100000000)#100megabit
#                 yield self.env.timeout(sending_time)
#                 self.aggregator.put(p)
#             off_period = self.off()/1000
#             self.env.timeout(off_period)



# class SelfSimilar(PacketGenerator):
#     """This class represents the self-similar packet generation process """
#     def __init__(self,env, id, on_dist, off_dist, fix_pkt_size):
#         PacketGenerator.__init__(self,env, id)
#         self.SubStreamAggregator = simpy.Store(env)# sub-streams traffic aggregator
#
#
#     def run(self):
#         """The generator function used in simulations.
#         """
#         while self.env.now < self.finish:

class ONUPort(object):

    def __init__(self, env, qlimit=None):
        self.buffer = simpy.Store(env)#buffer
        self.grant_real_usage = simpy.Store(env) # Used in grant prediction report
        self.grant_size = 0
        self.grant_final_time = 0
        self.predicted_grant = False #flag if it is a predicted grant
        self.guard_interval = 0.000001
        self.env = env
        self.out = None # ONU port output
        self.packets_rec = 0 #received pkt counter
        self.packets_drop = 0#dropped pkt counter
        self.qlimit = qlimit #Buffer queue limit
        self.byte_size = 0  # Current size of the buffer in bytes
        self.busy = 0  # Used to track if a packet is currently being sent
        self.action = env.process(self.run())  # starts the run() method as a SimPy process
        self.pkt = None #network packet obj
        self.grant_loop = False #flag if grant time is being used
        self.current_grant_delay = []

    def get_current_grant_delay(self):
        return self.current_grant_delay
    def reset_curret_grant_delay(self):
        self.current_grant_delay = []

    def set_grant(self,grant,pred=False): #setting grant byte size and its ending
        self.grant_size = grant['grant_size']
        self.grant_final_time = grant['grant_final_time']
        self.predicted_grant = pred

    def get_pkt(self):
        """process to get the packet from the buffer   """

        try:
            pkt = (yield self.buffer.get() )#getting a packet from the buffer
            self.pkt = pkt

        except simpy.Interrupt as i:
            logging.debug("Error while getting a packet from the buffer ({})".format(i))

            pass

        if not self.grant_loop:#put the pkt back to the buffer if the grant time expired

            self.buffer.put(pkt)



    def send(self,ONU_id):
        """ process to send pkts
        """
        self.grant_loop = True #flag if grant time is being used
        start_grant_usage = None #grant timestamp
        end_grant_usage = 0 #grant timestamp
        why_break = "ok"

        #self.current_grant_delay = []

        while self.grant_final_time > self.env.now:

            get_pkt = self.env.process(self.get_pkt())#trying to get a package in the buffer
            grant_timeout = self.env.timeout(self.grant_final_time - self.env.now)
            yield get_pkt | grant_timeout#wait for a package to be sent or the grant timeout

            if (self.grant_final_time <= self.env.now):
                #The grant time has expired
                why_break ="time expired"
                break
            if self.pkt is not None:
                pkt = self.pkt
                if not start_grant_usage:
                    start_grant_usage = self.env.now #initialized the real grant usage time
                start_pkt_usage = self.env.now ##initialized the pkt usage time

            else:
                #there is no pkt to be sent
                logging.debug("{}: there is no packet to be sent".format(self.env.now))
                why_break = "no pkt"
                break
            self.busy = 1
            self.byte_size -= pkt.size
            if self.byte_size < 0:#Prevent the buffer from being negative
                logging.debug("{}: Negative buffer".format(self.env.now))
                self.byte_size += pkt.size
                self.buffer.put(pkt)
                why_break = "negative buffer"
                break

            bits = pkt.size * 8
            sending_time = 	bits/float(10000000000) # buffer transmission time

            #To avoid fragmentation by passing the Grant window
            if env.now + sending_time > self.grant_final_time + self.guard_interval:
                self.byte_size += pkt.size

                self.buffer.put(pkt)
                why_break = "fragmentation"
                break

            #write the pkt transmission delay
            delay_file.write( "{},{}\n".format( ONU_id, (self.env.now - pkt.time) ) )
            self.current_grant_delay.append(self.env.now - pkt.time)
            if self.predicted_grant:
                delay_prediction_file.write( "{},{}\n".format( ONU_id, (self.env.now - pkt.time) ) )

            else:
                delay_normal_file.write( "{},{}\n".format( ONU_id, (self.env.now - pkt.time) ) )

            yield self.env.timeout(sending_time)

            end_pkt_usage = self.env.now
            end_grant_usage += end_pkt_usage - start_pkt_usage

            self.pkt = None

        #ending of the grant
        self.grant_loop = False #flag if grant time is being used
        if start_grant_usage and end_grant_usage > 0:# if any pkt has been sent
            #send the real grant usage
            self.grant_real_usage.put( [start_grant_usage , start_grant_usage + end_grant_usage] )
        else:
            #logging.debug("buffer_size:{}, grant duration:{}".format(b,grant_timeout))
            print why_break
            self.grant_real_usage.put([])# send a empty list



    def run(self): #run the port as a simpy process
        while True:
            yield self.env.timeout(5)


    def put(self, pkt):
        """receives a packet from the packet genarator and put it on the queue
            if the queue is not full, otherwise drop it.
        """

        self.packets_rec += 1
        tmp = self.byte_size + pkt.size
        if self.qlimit is None: #checks if the queue size is unlimited
            self.byte_size = tmp
            return self.buffer.put(pkt)
        if tmp >= self.qlimit: # chcks if the queue is full
            self.packets_drop += 1
            #return
        else:
            self.byte_size = tmp
            self.buffer.put(pkt)

class ONU(object):
    def __init__(self,distance,oid,env,lamb,odn,qlimit,bucket,packet_gen,pg_param):
        self.env = env
        self.odn= odn
        self.grant_report_store = simpy.Store(self.env) #Simpy Stores grant usage report
        self.request_container = simpy.Container(env, init=2, capacity=2)
        self.grant_report = []
        self.distance = distance #fiber distance
        self.oid = oid #ONU indentifier
        self.delay = self.distance/ float(210000) # fiber propagation delay
        self.excess = 0 #difference between the size of the request and the grant
        self.newArrived = 0
        self.last_req_buffer = 0
        self.request_counter = 0
        self.pg = packet_gen(self.env, "bbmp", self, **pg_param) #creates the packet generator
        if qlimit == 0:# checks if the queue has a size limit
            queue_limit = None
        else:
            queue_limit = qlimit
        self.port = ONUPort(self.env, qlimit=queue_limit)#create ONU PORT
        self.pg.out = self.port #forward packet generator output to ONU port
        if not (DBA_ALGORITHM == "mpd_dba" or DBA_ALGORITHM == "mdba"):
            self.sender = self.env.process(self.ONU_sender(odn))
        self.receiver = self.env.process(self.ONU_receiver(odn))
        self.bucket = bucket #Bucket size
        self.lamb = lamb # wavelength lambda


    def ONU_receiver(self,odn):
        while True:
            # Grant stage
            grant = yield odn.get_grant(self.oid)#waiting for a grant
            pred_grant_usage_report = [] # grant prediction report list
            # real start and endtime used report to OLT

            try:
                # print grant['grant_final_time']
                # print self.oid
                next_grant = grant['grant_start_time'] - self.env.now #time until next grant begining
                yield self.env.timeout(next_grant)  #wait for the next grant
            except Exception as e:
                pass
                #print grant['grant_start_time']
                #print self.env.now

            self.excess = self.last_req_buffer - grant['grant_size'] #update the excess
            self.port.set_grant(grant,False) #grant info to onu port

            sent_pkt = self.env.process(self.port.send(self.oid)) # send pkts during grant time
            yield sent_pkt # wait grant be used
            grant_usage = yield self.port.grant_real_usage.get() # get grant real utilisation
            if len(grant_usage) == 0: #debug
                logging.debug("Error in grant_usage")
            #yield self.env.timeout(self.delay)

            # Prediction stage
            if grant['prediction']:#check if have any predicion in the grant

                self.port.reset_curret_grant_delay()
                for pred in grant['prediction']:
                    # construct grant pkt
                    pred_grant = {'grant_size': grant['grant_size'], 'grant_final_time': pred[1]}
                    #wait next cycle
                    try:
                        next_grant = pred[0] - self.env.now #time until next grant begining
                        yield self.env.timeout(next_grant)  #wait for the next grant
                    except Exception as e:
                        logging.debug("{}: pred {}, gf {}".format(self.env.now,pred,grant['grant_final_time']))
                        logging.debug("Error while waiting for the next grant ({})".format(e))
                        break

                    self.port.set_grant(pred_grant,True) #grant info to onu port
                    sent_pkt = self.env.process(self.port.send(self.oid))#sending predicted messages
                    yield sent_pkt # wait grant be used
                    grant_usage = yield self.port.grant_real_usage.get() # get grant real utilisation
                    yield self.env.timeout(self.delay) # wait grant propagation delay
                    if len(grant_usage) > 0: # filling grant prediction report list
                        pred_grant_usage_report.append(grant_usage)
                        #logging.debug("{}:pred={},usage={}".format(self.env.now,pred,grant_usage))
                    else:
                        logging.debug("{}:Error in pred_grant_usage".format(self.env.now))
                        break
            # grant mean squared errors
            # if len(pred_grant_usage_report) > 0 and len(pred_grant_usage_report) != len(grant['prediction']):
            #     logging.debug("{}:Error predictions len is diff of pred usage ({})".format(self.env.now, len(grant['prediction']) - len(pred_grant_usage_report) ))
            if len(pred_grant_usage_report) > 0:
                delay = self.port.get_current_grant_delay()
                if len(delay) == 0:
                    logging.debug("{}:Error in current grant delay".format(self.env.now))
                    # print pred_grant_usage_report
                    # print delay
                    delay.append(-1)
                len_usage = len(pred_grant_usage_report)
                mse_start = mse(np.array(pred_grant_usage_report)[:,0],np.array(grant['prediction'][:len_usage])[:,0])
                mse_end = mse(np.array(pred_grant_usage_report)[:,1],np.array(grant['prediction'][:len_usage])[:,1])
                mse_file.write("{},{},{}\n".format(mse_start,mse_end,np.mean(delay)))
            self.port.reset_curret_grant_delay()
            yield self.env.timeout(self.delay) # propagation delay

            #Signals the end of grant processing to allow new requests
            if not (DBA_ALGORITHM == "mpd_dba" or DBA_ALGORITHM == "mdba"):
                yield self.grant_report_store.put(pred_grant_usage_report)
################################################################
    #IPACT
    def ONU_sender(self, odn):
        """A process which checks the queue size and send a REQUEST message to OLT"""
        while True:
            # send a REQUEST only if the queue size is greater than the bucket size
            #yield self.request_container.get(1)
            if self.port.byte_size >= self.bucket:
                requested_buffer = self.port.byte_size #gets the size of the buffer that will be requested
                #update the size of the current/last buffer REQUEST
                self.last_req_buffer = requested_buffer
                # creating request message
                msg = {'text':"ONU %s sent this REQUEST for %.6f at %f" %
                    (self.oid,self.port.byte_size, self.env.now),'buffer_size':requested_buffer,'ONU':self}
                odn.put_request((msg),self)# put the request message in the odn

                # Wait for the grant processing to send the next request
                self.grant_report = yield self.grant_report_store.get()
                #yield self.env.timeout(2*self.delay)
            else: # periodic check delay
                #yield self.request_container.put(1)
                yield self.env.timeout(self.delay)
    #MTP
    # def ONU_senderMT(self, odn):
    #     """A process which checks the queue size and send a REQUEST message to OLT"""
    #     while True:
    #         # send a REQUEST only if the queue size is greater than the bucket size
    #         yield self.request_container.get(1)
    #
    #         requested_buffer = self.port.byte_size #gets the size of the buffer that will be requested
    #         #update the size of the current/last buffer REQUEST
    #         self.newArrived = requested_buffer - self.last_req_buffer
    #         self.last_req_buffer = requested_buffer
    #         # creating request message
    #         msg = {'text':"ONU %s sent this REQUEST for %.6f at %f" %
    #             (self.oid,self.port.byte_size, self.env.now),'buffer_size':requested_buffer,'ONU':self}
    #         odn.put_request((msg),self)# put the request message in the odn
    #
    #         # Wait for the grant processing to send the next request
    #         #self.grant_report = yield self.grant_report_store.get()
    #         #yield self.env.timeout(2*self.delay)


class DBA(object):
    """DBA Parent class, heritated by every kind of DBA"""
    def __init__(self,env,max_grant_size,grant_store):
        self.env = env
        self.max_grant_size = max_grant_size
        self.grant_store = grant_store
        self.guard_interval = 0.000001

class IPACT(DBA):
    def __init__(self,env,max_grant_size,grant_store):
        DBA.__init__(self,env,max_grant_size,grant_store)
        self.counter = simpy.Resource(self.env, capacity=1)#create a queue of requests to DBA


    def dba(self,ONU,buffer_size):
        with self.counter.request() as my_turn:
            """ DBA only process one request at a time """
            yield my_turn
            time_stamp = self.env.now # timestamp dba starts processing the request
            delay = ONU.delay # oneway delay

            # check if max grant size is enabled
            if self.max_grant_size > 0 and buffer_size > self.max_grant_size:
                buffer_size = self.max_grant_size
            bits = buffer_size * 8
            sending_time = 	bits/float(10000000000) #buffer transmission time
            grant_time = delay + sending_time
            grant_final_time = self.env.now + grant_time # timestamp for grant end
            counter = Grant_ONU_counter[ONU.oid] # Grant message counter per ONU
            # write grant log
            grant_time_file.write( "{},{},{},{},{},{},{},{}\n".format(MAC_TABLE['olt'], MAC_TABLE[ONU.oid],"02", time_stamp,counter, ONU.oid,self.env.now,grant_final_time) )
            # construct grant message
            grant = {'ONU':ONU,'grant_size': buffer_size, 'grant_start_time': self.env.now , 'grant_final_time': grant_final_time, 'prediction': None}
            self.grant_store.put(grant) # send grant to OLT
            Grant_ONU_counter[ONU.oid] += 1

            # timeout until the end of grant to then get next grant request
            yield self.env.timeout(delay+grant_time + self.guard_interval)

class MDBA(DBA):
    def __init__(self,env,max_grant_size,grant_store):
        DBA.__init__(self,env,max_grant_size,grant_store)
        self.counter = simpy.Resource(self.env, capacity=1)#create a queue of requests to DBA
        self.next_grant = 0


    def dba(self,ONU,buffer_size):
        with self.counter.request() as my_turn:
            """ DBA only process one request at a time """
            yield my_turn
            time_stamp = self.env.now # timestamp dba starts processing the request
            delay = ONU.delay # oneway delay
            yield self.env.timeout(self.guard_interval)

            # check if max grant size is enabled
            if self.max_grant_size > 0 and buffer_size > self.max_grant_size:
                buffer_size = self.max_grant_size
            bits = buffer_size * 8
            sending_time = 	bits/float(10000000000) #buffer transmission time
            grant_time = delay + sending_time
            ini = max(self.env.now,self.next_grant)
            grant_final_time = ini + grant_time # timestamp for grant end
            counter = Grant_ONU_counter[ONU.oid] # Grant message counter per ONU
            # write grant log
            grant_time_file.write( "{},{},{},{},{},{},{},{}\n".format(MAC_TABLE['olt'], MAC_TABLE[ONU.oid],"02", time_stamp,counter, ONU.oid,self.env.now,grant_final_time) )
            # construct grant message
            grant = {'ONU':ONU,'grant_size': buffer_size,'grant_start_time':ini ,'grant_final_time': grant_final_time, 'prediction': None}
            self.grant_store.put(grant) # send grant to OLT
            Grant_ONU_counter[ONU.oid] += 1

            # timeout until the end of grant to then get next grant request
            self.next_grant = grant_final_time + delay + self.guard_interval

class MTP(DBA):
    def __init__(self,env,max_grant_size,grant_store,numberONUs,NThreads=2):
        DBA.__init__(self,env,max_grant_size,grant_store)
        self.interTh_store = simpy.Store(self.env)
        self.ThreadList = []
        #create threads
        for i in range(NThreads):
            self.ThreadList.append(MTP_THREAD(env,i,numberONUs,self.guard_interval,max_grant_size,grant_store,self.interTh_store))
        self.currentThread = 0
        self.nextThread = 1
        self.ThreadManager_proc = self.env.process(self.ThreadManager())

    def dba(self,ONU,buffer_size):
        status = self.ThreadList[self.currentThread].getRequestStatus(ONU.oid)
        if status == 0:
            self.ThreadList[self.currentThread].request_store.put((ONU,buffer_size))
        else:
            self.ThreadList[self.nextThread].request_store.put((ONU,buffer_size))
    def ThreadManager(self):
        while True:
            msg = yield self.interTh_store.get()
            if msg['msg'] == 'getNextTHRequest':
                requestList = self.ThreadList[self.nextThread].getRequestList()
                self.ThreadList[self.currentThread].NextTHRequest_store.put(requestList)
            elif msg['msg'] == 'updateNextTHRequest':
                self.ThreadList[self.nextThread].updateRequest(msg['data'])
            elif msg['msg'] == 'endThread':
                self.ThreadList[self.nextThread].setCycleStart(msg['data'])
                aux = self.currentThread
                self.currentThread = self.nextThread
                self.nextThread = aux
            else:
                print "ESTA ERRADO"



class MTP_THREAD(object):
    def __init__(self,env,tNumber,numberONUs,guard_interval,Bmin,grant_store,interTh_store):
        self.env = env
        self.threadNumber = tNumber
        self.numberONUs = numberONUs
        self.guard_interval = guard_interval
        self.Bmin = Bmin
        self.request_counter = 0
        self.requestList = []
        self.grantList = []
        for i in range(self.numberONUs):
            self.requestList.append({'bandw':0,'ONU':None,'buffer_size':0})
        self.excess = 0
        self.lowLoadList = [] #ONU_id,
        self.highLoadList = [] #tuple ONU_id, excess
        self.cycleStart = self.env.now
        #self.cycleEnd = self.cycleStart + self.Bmin*(self.numberONUs)
        self.grant_store = grant_store
        self.interTh_store = interTh_store #sends msg to ThreaddbaManager
        self.request_store = simpy.Store(self.env) #receives request from ThreaddbaManager
        self.NextTHRequest_store = simpy.Store(self.env) #receives NextTHRequestList from ThreaddbaManager
        self.reqGathering_ends = self.env.event()
        self.RequestManager_proc = self.env.process(self.RequestManager())
        self.dba_proc = self.env.process(self.dba())

    def updateRequest(self,requestList):
        for req in requestList:
            self.requestList[req[0]]['status'] = 1
            self.requestList[req[0]]['buffer_size'] += req[1]
    def getRequestStatus(self,ONU_id):
        return self.requestList[ONU_id]['status']
    def getRequestList(self):
        return self.requestList
    def setCycleStart(self,start):
        self.cycleStart = start
    def getCycleStart(self):
        return self.cycleStart
    # def setCycleEnd(self,start):
    #     self.cycleEnd = end
    # def getCycleEnd(self):
    #     return self.cycleEnd

    def RequestManager(self):
        while True:
            ONU,buffer_size = yield self.request_store.get()
            if self.requestList[ONU.oid]['buffer_size'] == 0:
                #self.requestList[ONU.oid]['status'] = 1
                self.requestList[ONU.oid]['ONU'] = ONU
                self.requestList[ONU.oid]['buffer_size'] = buffer_size
                self.request_counter+=1
            else:
                print "erro request repetido"
            if self.request_counter == self.numberONUs:
                self.reqGathering_ends.succeed()
                self.reqGathering_ends = self.env.event()
    def dba(self):
        yield self.reqGathering_ends
        #calculate real bandwith demand
        for oid,req in enumerate(self.requestList):
            Newtraffic = self.lastTHRequestList[oid]['buffer_size'] - req['buffer_size']
            self.requestList[oid]['bandw'] = Newtraffic + self.lastTHRequestList[oid]['backlogged']

        #check valid requests in next thread
        self.interTh_store.put({'threadNumber':self.threadNumber,'msg':'getNextTHRequest','data':None})
        NextTHRequest = yield self.NextTHRequest_store.get()
        updateNextTHRequestList = []
        for oid,req in enumerate(self.requestList):
            if NextTHRequest[oid]['buffer_size'] != 0:
                nextNewtraffic = NextTHRequest[oid]['buffer_size'] - req['buffer_size']
                self.requestList[oid]['bandw'] += nextNewtraffic
                nreq = self.requestList[oid]['buffer_size']
                updateNextTHRequestList.append([oid,nreq])

        # updateNextTHRequestList = []
        # for lowLoad in self.lowLoadList:
        #     if NextTHRequest[lowLoad[0]]['status'] == 1:
        #         bandw = NextTHRequest[lowLoad[0]]['buffer_size'] - lowLoad[1]
        #         if NextTHRequest[lowLoad[0]]['buffer_size'] <=0:
        #             print "ALGO MUITO ERRADO"
        #         if bandw >= 0:
        #             self.grantList.append([lowLoad[0],self.Bmin])
        #             self.excess -= lowLoad[1]
        #             updateNextTHRequestList.append([lowLoad[0],-1*(lowLoad[1])])
        #         else:
        #             self.grantList.append( [lowLoad[0],
        #                 self.requestList[lowLoad[0]]['buffer_size'] + NextTHRequest[lowLoad[0]]['buffer_size'] )
        #             self.excess -= NextTHRequest[lowLoad[0]]['buffer_size']
        #             updateNextTHRequestList.append([lowLoad[0],
        #                 -1*(NextTHRequest[lowLoad[0]]['buffer_size'])])
        # self.interTh_store.put({'threadNumber':self.threadNumber,'msg':'updateNextTHRequest','data':updateNextTHRequestList})
        #Split high load ONUs
        for oid,req in enumerate(self.requestList):
            bandw = self.Bmin - req['buffer_size']
            if bandw < 0:
                self.highLoadList.append([oid,-1*(bandw)])
            else:
                self.excess += bandw
                self.lowLoadList.append([oid,bandw])
        #distributing excess
        if self.excess < 0:
            print "DEU MERDA"
        if self.excess > 0:
            highloadbuffer = 0
            updateNextTHRequestList = []
            excessDistributionList = []
            for highload in self.highLoadList:
                highloadbuffer += highload[1]
            for highload in self.highLoadList:
                bandw = (highload[1]*self.excess)/float(highloadbuffer)
                excessDistributionList.append([highload[0],bandw])
            for i,excess_dist in enumerate(xcessDistributionList):
                self.grantList.append([excess_dist[0],
                    self.Bmin+excess_dist[1]])
                self.updateNextTHRequestList([excess_dist[0],
                        highLoadList[i][1]-excess_dist[1]])
            self.interTh_store.put({'threadNumber':self.threadNumber,'msg':'updateNextTHRequest','data':updateNextTHRequestList})
            if self.excess != 0:
                print "WE HAVE A PROBLEM"
        #Sending grants
        if self.env.now >= self.cycleStart:
            next_time = self.env.now
        else:
            next_time = self.cycleStart
        for onu_grant in self.grantList:
            ONU = self.requestList[onu_grant[0]]["ONU"]
            delay = ONU.delay
            bits = onu_grant[1] * 8
            sending_time = 	bits/float(10000000000) #buffer transmission time
            grant_time = delay + sending_time
            grant_final_time = next_time + grant_time # timestamp for grant end
            # construct grant message
            grant = {'ONU':ONU,'grant_size': onu_grant[1], 'grant_final_time': grant_final_time, 'prediction': None}
            self.grant_store.put(grant) # send grant to OLT
            Grant_ONU_counter[ONU.oid] += 1

            # next ONU grant start time
            next_time = delay+grant_time + self.guard_interval
            self.requestList[onu_grant[0]]['status'] = 0
            self.requestList[onu_grant[0]]['buffer_size'] = 0
        self.interTh_store.put({'threadNumber':self.threadNumber,'msg':'endThread','data':next_time})



class PD_DBA(DBA):
    def __init__(self,env,max_grant_size,grant_store,window=20,predict=5,model="ols"):
        DBA.__init__(self,env,max_grant_size,grant_store)
        self.counter = simpy.Resource(self.env, capacity=1)#create a queue of requests to DBA
        self.window = window    # past observations window size
        self.predict = predict # number of predictions
        self.grant_history = range(NUMBER_OF_ONUs) #grant history per ONU (training set)
        self.predictions_array = []
        for i in range(NUMBER_OF_ONUs):
            # training unit
            self.grant_history[i] = {'counter': [], 'start': [], 'end': []}
        #Implementing the model
        if model == "ols":
            reg = linear_model.LinearRegression()
        else:
            reg = linear_model.Ridge(alpha=.5)

        self.model = MultiOutputRegressor(reg)

    def predictions_schedule(self,predictions):
        predictions = map(list,predictions)
        predictions_cp = list(predictions)
        if len(self.predictions_array) > 0:
            self.predictions_array = filter(lambda x: x[0] > self.env.now, self.predictions_array)
            # for interval1 in predictions:
            #     for interval2 in self.predictions_array:
            #         if interval2[1] > interval1[0]:
            #             # print predictions
            #             index = predictions.index(interval1)
            #             new_interval = [ interval2[1] , interval1[1] ]
            #             predictions_cp[ index ] = new_interval
            #         elif interval1[1] > interval2[0]:
            #             index = predictions.index(interval1)
            #             new_interval = [ interval1[0] , interval2[0] ]
            #             predictions_cp[ index ] = new_interval
            # print predictions_cp
            # time.sleep(5)
            # predictions = predictions_cp



        # print self.predictions_array
        # print ""
        predictions_array_cp = list(self.predictions_array)
        predictions_array_cp +=  predictions
        predictions_array_cp.sort()
        #self.predictions_array = sorted(self.predictions_array,key=lambda x: x[0])
        # print self.predictions_array
        # print "#########"
        # time.sleep(5)

        #self.predictions_array.sort()
        over = False
        j = 1
        for interval1 in predictions_array_cp[:-1]:
            for interval2 in predictions_array_cp[j:]:
                if interval1[1] > interval2[0]:
                    overlap_file.write("{}\n".format(interval1[1] - interval2[0]))
                    over = True
                    if interval1 in predictions:
                        #index1 = self.predictions_array.index(interval1)
                        index = predictions.index(interval1)
                        new_interval = [ interval1[0] , interval2[0]]
                        predictions_cp[ index ] = new_interval
                        #self.predictions_array[index1] = new_interval

                    elif interval2 in predictions:
                        #index1 = self.predictions_array.index(interval2)
                        index = predictions.index(interval2)
                        new_interval = [ interval1[1], interval2[1] ]
                        predictions_cp[ index ] = new_interval
                        #self.predictions_array[index1] = new_interval
                else:
                    break
            j+=1

        if over:
            predictions = None
        else:
            predictions = predictions_cp
            self.predictions_array += predictions
        return predictions

    def drop_overlap(self,predictions,ONU):
        predcp = list(predictions)
        j = 1
        #drop: if there are overlaps between the predictions
        for p in predcp[:-1]:
            for q in predcp[j:]:
                if p[1] + ONU.delay  > q[0]:
                    predictions = None
                    break

            j+=1
        #drop: if there is overlap between standard grant and first prediction
        if predictions is not None and (self.grant_history[ONU.oid]['end'][-1] +ONU.delay+ self.guard_interval) > predictions[0][0]:
            predictions = None


        return predictions


    def predictor(self, ONU_id):
        # check if there's enough observations to fill window

        if len( self.grant_history[ONU_id]['start'] ) >= self.window :
            #reduce the grant history to the window size
            self.grant_history[ONU_id]['start'] = self.grant_history[ONU_id]['start'][-self.window:]
            self.grant_history[ONU_id]['end'] = self.grant_history[ONU_id]['end'][-self.window:]
            self.grant_history[ONU_id]['counter'] = self.grant_history[ONU_id]['counter'][-self.window:]
            df_tmp = pd.DataFrame(self.grant_history[ONU_id]) # temp dataframe w/ past grants
            # create a list of the next p Grants that will be predicted
            X_pred = np.arange(self.grant_history[ONU_id]['counter'][-1] +1, self.grant_history[ONU_id]['counter'][-1] + 1 + self.predict).reshape(-1,1)

            # model fitting
            self.model.fit( np.array( df_tmp['counter'] ).reshape(-1,1) , df_tmp[['start','end']] )
            pred = self.model.predict(X_pred) # predicting start and end

            predictions = list(pred)
            #predictions = self.predictions_schedule(predictions)

            return predictions

        else:
            return  None


    def dba(self,ONU,buffer_size):
        with self.counter.request() as my_turn:
            """ DBA only process one request at a time """
            yield my_turn
            time_stamp = self.env.now
            delay = ONU.delay

            if len(ONU.grant_report) > 0:
                # if predictions where utilized, update history with real grant usage
                for report in ONU.grant_report:
                    self.grant_history[ONU.oid]['start'].append(report[0])
                    self.grant_history[ONU.oid]['end'].append(report[1])
                    self.grant_history[ONU.oid]['counter'].append( self.grant_history[ONU.oid]['counter'][-1] + 1  )

            # check if max grant size is enabled
            if self.max_grant_size > 0 and buffer_size > self.max_grant_size:
                buffer_size = self.max_grant_size
            bits = buffer_size * 8
            sending_time = 	bits/float(10000000000) #buffer transmission time10g
            grant_time = delay + sending_time # one way delay + transmission time
            grant_final_time = self.env.now + grant_time # timestamp for grant end

            # Update grant history with grant requested
            self.grant_history[ONU.oid]['start'].append(self.env.now)
            self.grant_history[ONU.oid]['end'].append(grant_final_time)
            if len(self.grant_history[ONU.oid]['counter']) > 0:
                self.grant_history[ONU.oid]['counter'].append( self.grant_history[ONU.oid]['counter'][-1] + 1  )
            else:
                self.grant_history[ONU.oid]['counter'].append( 1 )

            #PREDICTIONS
            predictions = self.predictor(ONU.oid) # start predictor process

            #drop if the predictions have overlap
            if predictions is not None:
                predictions = self.drop_overlap(predictions,ONU)


            #grant_time_file.write( "{},{},{}\n".format(ONU.oid,self.env.now,grant_final_time) )
            # construct grant message
            grant = {'ONU':ONU,'grant_size': buffer_size, 'grant_start_time': self.env.now , 'grant_final_time': grant_final_time, 'prediction': predictions}

            self.grant_store.put(grant) # send grant to OLT

            # timeout until the end of grant to then get next grant request
            yield self.env.timeout(grant_time+delay+ self.guard_interval)


class MPD_DBA(DBA):
    def __init__(self,env,max_grant_size,grant_store,window=20,predict=5,model="ols"):
        DBA.__init__(self,env,max_grant_size,grant_store)
        self.counter = simpy.Resource(self.env, capacity=1)#create a queue of requests to DBA
        self.window = window    # past observations window size
        self.predict = predict # number of predictions
        self.next_grant = 0
        self.grant_history = range(NUMBER_OF_ONUs) #grant history per ONU (training set)
        self.predictions_array = []
        self.predictions_counter_array = []
        for i in range(NUMBER_OF_ONUs):
            # training unit
            self.grant_history[i] = {'counter': [], 'start': [], 'end': []}
            self.predictions_counter_array.append(0)
        #Implementing the model
        if model == "ols":
            reg = linear_model.LinearRegression()
        else:
            reg = linear_model.Ridge(alpha=.5)

        self.model = MultiOutputRegressor(reg)

    def drop_overlap(self,predictions,ONU):
        predcp = list(predictions)
        j = 1
        #drop: if there are overlaps between the predictions
        for p in predcp[:-1]:
            for q in predcp[j:]:
                if p[1] + ONU.delay  > q[0]:
                    predictions = None
                    break

            j+=1

        #drop: if there is overlap between standard grant and first prediction
        if predictions is not None and (self.grant_history[ONU.oid]['end'][-1] +ONU.delay+ self.guard_interval) > predictions[0][0]:
            predictions = None


        return predictions

    def predictor(self, ONU_id):
        # check if there's enough observations to fill window

        if len( self.grant_history[ONU_id]['start'] ) >= self.window :
            #reduce the grant history to the window size
            self.grant_history[ONU_id]['start'] = self.grant_history[ONU_id]['start'][-self.window:]
            self.grant_history[ONU_id]['end'] = self.grant_history[ONU_id]['end'][-self.window:]
            self.grant_history[ONU_id]['counter'] = self.grant_history[ONU_id]['counter'][-self.window:]
            df_tmp = pd.DataFrame(self.grant_history[ONU_id]) # temp dataframe w/ past grants
            # create a list of the next p Grants that will be predicted
            X_pred = np.arange(self.grant_history[ONU_id]['counter'][-1] +1, self.grant_history[ONU_id]['counter'][-1] + 1 + self.predict).reshape(-1,1)

            # model fitting
            self.model.fit( np.array( df_tmp['counter'] ).reshape(-1,1) , df_tmp[['start','end']] )
            pred = self.model.predict(X_pred) # predicting start and end

            predictions = list(pred)
            #predictions = self.predictions_schedule(predictions)

            return predictions

        else:
            return  None


    def dba(self,ONU,buffer_size):
        with self.counter.request() as my_turn:
            """ DBA only process one request at a time """
            yield my_turn
            time_stamp = self.env.now
            delay = ONU.delay
            yield self.env.timeout(self.guard_interval)
            if len(ONU.grant_report) > 0:
                # if predictions where utilized, update history with real grant usage
                for report in ONU.grant_report:
                    self.grant_history[ONU.oid]['start'].append(report[0])
                    self.grant_history[ONU.oid]['end'].append(report[1])
                    self.grant_history[ONU.oid]['counter'].append( self.grant_history[ONU.oid]['counter'][-1] + 1  )

            # check if max grant size is enabled
            if self.max_grant_size > 0 and buffer_size > self.max_grant_size:
                buffer_size = self.max_grant_size
            bits = buffer_size * 8
            sending_time = 	bits/float(10000000000) #buffer transmission time10g
            grant_time = delay + sending_time # one way delay + transmission time
            ini = max(self.env.now,self.next_grant)
            grant_final_time = ini + grant_time # timestamp for grant end

            # Update grant history with grant requested
            self.grant_history[ONU.oid]['start'].append(ini)
            self.grant_history[ONU.oid]['end'].append(grant_final_time)
            if len(self.grant_history[ONU.oid]['counter']) > 0:
                self.grant_history[ONU.oid]['counter'].append( self.grant_history[ONU.oid]['counter'][-1] + 1  )
            else:
                self.grant_history[ONU.oid]['counter'].append( 1 )

            if self.predictions_counter_array[ONU.oid] > 0:
                self.predictions_counter_array[ONU.oid] -= 1
            else:
                #PREDICTIONS
                predictions = self.predictor(ONU.oid) # start predictor process

                #drop if the predictions have overlap
                # if predictions is not None:
                #     predictions = self.drop_overlap(predictions,ONU)

                if predictions is not None:
                    self.predictions_counter_array[ONU.oid] = len(predictions)


                #grant_time_file.write( "{},{},{}\n".format(ONU.oid,self.env.now,grant_final_time) )
                # construct grant message
                grant = {'ONU':ONU,'grant_size': buffer_size,'grant_start_time':ini , 'grant_final_time': grant_final_time, 'prediction': predictions}

                self.grant_store.put(grant) # send grant to OLT

                # timeout until the end of grant to then get next grant request
                self.next_grant = grant_final_time + delay + self.guard_interval



class OLT(object):
    """Optical line terminal"""
    def __init__(self,env,lamb,odn,max_grant_size,dba,window,predict,model,numberONUs):
        self.env = env
        self.lamb = lamb
        self.grant_store = simpy.Store(self.env) # grant communication between processes
        #choosing algorithms
        if dba == "pd_dba":
            self.dba = PD_DBA(self.env, max_grant_size, self.grant_store,window,predict,model)
        elif dba == "mpd_dba":
            self.dba = MPD_DBA(self.env, max_grant_size, self.grant_store,window,predict,model)
        elif dba == "mdba":
            self.dba = MDBA(self.env, max_grant_size, self.grant_store)
        else:
            self.dba = IPACT(self.env, max_grant_size, self.grant_store)

        self.receiver = self.env.process(self.OLT_receiver(odn)) # process for receiving requests
        self.sender = self.env.process(self.OLT_sender(odn)) # process for sending grant

    def OLT_sender(self,odn):
        """A process which sends a grant message to ONU"""
        while True:
            grant = yield self.grant_store.get() # receive grant from dba
            odn.put_grant(grant['ONU'],grant) # send grant to odn

    def OLT_receiver(self,odn):
        """A process which receives a request message from the ONUs."""
        while True:
            request = yield odn.get_request(self.lamb) #get a request message
            #print("Received Request from ONU {} at {}".format(request['ONU'].oid, self.env.now))
            # send request to DBA
            self.env.process(self.dba.dba(request['ONU'],request['buffer_size']))


#starts the simulator environment
random.seed(RANDOM_SEED)
env = simpy.Environment()

#creates the optical distribution network
odn = ODN(env)

#Packet generator
if TRAFFIC == "poisson":
    packet_gen = poisson_PG
    pg_param = {"adist":functools.partial(random.expovariate, EXPONENT), "sdist":None, "fix_pkt_size":PKT_SIZE}
else:
    packet_gen = CBR_PG
    pg_param = {"fix_pkt_size":PKT_SIZE}


#Creates the ONUs
ONU_List = []
#lambda esta improvisado aqui criar por argumento
lamb = 0
for i in range(NUMBER_OF_ONUs):
    MAC_TABLE[i] = "00:00:00:00:{}:{}".format(random.randint(0x00, 0xff),random.randint(0x00, 0xff))
    Grant_ONU_counter[i] = 0
MAC_TABLE['olt'] = "ff:ff:ff:ff:00:01"
for i in range(NUMBER_OF_ONUs):
    distance= DISTANCE
    ONU_List.append(ONU(distance,i,env,lamb,odn,ONU_QUEUE_LIMIT,MAX_BUCKET_SIZE,packet_gen,pg_param))

#creates OLT
olt = OLT(env,lamb,odn,MAX_GRANT_SIZE,DBA_ALGORITHM,WINDOW,PREDICT,MODEL,NUMBER_OF_ONUs)
logging.info("starting simulator")
env.run(until=SIM_DURATION)
delay_file.close()
delay_normal_file.close()
delay_prediction_file.close()
grant_time_file.close()
pkt_file.close()
mse_file.close
overlap_file.close()

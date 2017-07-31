import simpy
import random
import functools
import time
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
import logging
from sklearn import linear_model


#Parsing the inputs arguments
parser = argparse.ArgumentParser(description="Long Reach PON Simulator")
group = parser.add_mutually_exclusive_group()
#group.add_argument("-v", "--verbose", action="store_true")
group.add_argument("-q", "--quiet", action="store_true")
parser.add_argument("N", type=int, default=3, help="the number of ONUs")
parser.add_argument("B", type=int, default=0, nargs='?', help="the size of the ONU sender bucket in bytes")
parser.add_argument("Q", type=int, default=None, nargs='?',help="the size of the ONU port queue in bytes")
parser.add_argument("M", type=float, default=0, nargs='?', help="The maximum size of buffer which a grant can allow")
parser.add_argument("D", type=int, default=100, nargs='?', help="Distance in km from ONU to OLT")
args = parser.parse_args()

#Arguments
NUMBER_OF_ONUs= args.N
DISTANCE = args.D
MAX_GRANT_SIZE = args.M
MAX_BUCKET_SIZE = args.B
ONU_QUEUE_LIMIT = args.Q

#settings
SIM_DURATION = 30
RANDOM_SEED = 20
PKT_SIZE = 9000
MAC_TABLE = {}
Grant_ONU_counter = {}

#logging
logging.basicConfig(filename='g-sim.log',level=logging.DEBUG,format='%(asctime)s %(message)s')
delay_file = open("delay.csv","w")
delay_file.write("ONU_id,delay\n")
grant_time_file = open("grant_time.csv","w")
grant_time_file.write("source address,destination address,opcode,timestamp,counter,ONU_id,start,end\n")
pkt_file = open("pkt.csv","w")
pkt_file.write("size\n")

class Cable(object):
    """This class represents the propagation through a cable and the splitter."""
    def __init__(self, env):
        self.env = env
        self.upstream = simpy.Store(env)
        self.downstream = []
        for i in range(NUMBER_OF_ONUs):
            self.downstream.append(simpy.Store(env))

    def up_latency(self, value,delay):
        """Calculates upstream propagation delay."""
        yield self.env.timeout(delay)
        self.upstream.put(value)

    def down_latency(self,ONU,value):
        """Calculates downstream propagation delay."""
        yield self.env.timeout(ONU.delay)
        self.downstream[ONU.oid].put(value)


    def put_request(self, value,delay):
        self.env.process(self.up_latency(value,delay))

    def get_request(self):
        return self.upstream.get()

    def put_grant(self,ONU,value):
        self.env.process(self.down_latency(ONU,value))

    def get_grant(self,ONU_id):
        return self.downstream[ONU_id].get()


class Packet(object):

    def __init__(self, time, size, id, src="a", dst="z", flow_id=0):
        self.time = time
        self.size = size
        self.id = id
        self.src = src
        self.dst = dst
        self.flow_id = flow_id

    def __repr__(self):
        return "id: {}, src: {}, time: {}, size: {}".\
            format(self.id, self.src, self.time, self.size)

class PacketGenerator(object):
    def __init__(self, env, id,  adist, sdist, fix_pkt_size=None, finish=float("inf"), flow_id=0):
        self.id = id
        self.env = env
        self.arrivals_dist = adist #packet arrivals distribution
        self.size_dist = sdist #packet size distribution

        self.fix_pkt_size = fix_pkt_size
        self.finish = finish
        self.out = None
        self.packets_sent = 0
        self.action = env.process(self.run())  # starts the run() method as a SimPy process
        self.flow_id = flow_id



    def run(self):
        """The generator function used in simulations.
        """


        while self.env.now < self.finish:
            # wait for next transmission
            yield self.env.timeout(self.arrivals_dist())
            self.packets_sent += 1


            if self.fix_pkt_size:
                p = Packet(self.env.now, self.fix_pkt_size, self.packets_sent, src=self.id, flow_id=self.flow_id)
                pkt_file.write("{}\n".format(self.fix_pkt_size))
            else:
                size = self.size_dist()
                p = Packet(self.env.now, size, self.packets_sent, src=self.id, flow_id=self.flow_id)
                pkt_file.write("{}\n".format(size))
            self.out.put(p)

class ONUPort(object):


    def __init__(self, env, qlimit=None):
        self.buffer = simpy.Store(env)#buffer
        self.grant_real_usage = simpy.Store(env)
        self.grant_size = 0
        self.grant_final_time = 0
        self.guard_interval = 0.000001
        self.env = env
        self.out = None
        self.packets_rec = 0
        self.packets_drop = 0
        self.qlimit = qlimit
        self.byte_size = 0  # Current size of the buffer in bytes
        self.last_buffer_size = 0 # size of the last buffer request
        self.busy = 0  # Used to track if a packet is currently being sent
        self.action = env.process(self.run())  # starts the run() method as a SimPy process
        self.pkt = None
        self.grant_loop = False




    def set_grant(self,grant): #setting grant byte size and its ending
        self.grant_size = grant['grant_size']
        self.grant_final_time = grant['grant_final_time']

    def update_last_buffer_size(self,requested_buffer): #update the size of the last buffer request
        self.last_buffer_size = requested_buffer

    def get_last_buffer_size(self): #return the size of the last buffer request
        return self.last_buffer_size

    def get_pkt(self):

        try:

            pkt = (yield self.buffer.get() )#getting a packet from the buffer


            self.pkt = pkt

        except simpy.Interrupt as i:
            logging.debug("Error while getting a packet from the buffer ({})".format(i))

            pass

        if not self.grant_loop:#put the pkt back to the buffer if the grant time expired

            self.buffer.put(pkt)



    def sent(self,ONU_id):
        self.grant_loop = True
        start_grant_usage = None
        current_grant_usage = 0
        while self.grant_final_time > self.env.now:

            get_pkt = self.env.process(self.get_pkt())#trying to get a package in the buffer
            grant_timeout = self.env.timeout(self.grant_final_time - self.env.now)
            yield get_pkt | grant_timeout#wait for a package to be sent or the grant timeout

            if (self.grant_final_time <= self.env.now):
                #The grant time has expired
                break
            if self.pkt is not None:
                pkt = self.pkt
                if not start_grant_usage:
                    start_grant_usage = self.env.now
                start_pkt_usage = self.env.now

            else:
                #there is not packate to be sent
                logging.debug("there is not packate to be sent")
                break
            self.busy = 1
            self.byte_size -= pkt.size
            if self.byte_size < 0:#Prevent the buffer from being negative

                self.byte_size += pkt.size
                self.buffer.put(pkt)
                break

            bits = pkt.size * 8
            sending_time = 	bits/float(1000000000)
            #To avoid fragmentation by passing the Grant window
            if env.now + sending_time > self.grant_final_time + self.guard_interval:
                self.byte_size += pkt.size

                self.buffer.put(pkt)
                break

            delay_file.write( "{},{}\n".format( ONU_id, (self.env.now - pkt.time) ) )
            yield self.env.timeout(sending_time)
            end_pkt_usage = self.env.now
            current_grant_usage += end_pkt_usage - start_pkt_usage

            self.pkt = None
        #ending of the grant
        self.grant_loop = False
        if start_grant_usage:
            self.grant_real_usage.put( [start_grant_usage , start_grant_usage + current_grant_usage] )
        else:
            self.grant_real_usage.put([])



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
    def __init__(self,distance,oid,env,cable,exp,qlimit,fix_pkt_size,bucket):
        self.env = env
        self.grant_report_store = simpy.Store(self.env) #Stores grant usage report
        self.grant_report = []
        self.distance = distance #fiber distance
        self.oid = oid #ONU indentifier
        self.delay = self.distance/ float(210000) # fiber propagation delay
        self.excess = 0 #difference between the size of the request and the grant
        arrivals_dist = functools.partial(random.expovariate, exp) #packet arrival distribuition
        size_dist = functools.partial(random.expovariate, 0.1)  # packet size distribuition, mean size 100 bytes
        self.pg = PacketGenerator(self.env, "bbmp", arrivals_dist, size_dist,fix_pkt_size) #creates the packet generator
        if qlimit == 0:# checks if the queue has a size limit
            queue_limit = None
        else:
            queue_limit = qlimit
        self.port = ONUPort(self.env, qlimit=queue_limit)#create ONU PORT
        self.pg.out = self.port #forward packet generator output to ONU port
        self.sender = self.env.process(self.ONU_sender(cable))
        self.receiver = self.env.process(self.ONU_receiver(cable))
        self.bucket = bucket #Bucket size


    def ONU_receiver(self,cable):
        while True:
            grant = yield cable.get_grant(self.oid)#waiting for a grant
            pred_grant_usage_report = []

            self.excess = self.port.get_last_buffer_size() - grant['grant_size'] #update the excess
            self.port.set_grant(grant)

            sent_pkt = self.env.process(self.port.sent(self.oid))
            yield sent_pkt
            grant_usage = yield self.port.grant_real_usage.get()
            if len(grant_usage) == 0:
                logging.debug("Erro in grant_usage")

            if grant['prediction']:#check if have any predicion in the grant
                for pred in grant['prediction'][1:]:
                    pred_grant = {'grant_size': grant['grant_size'], 'grant_final_time': pred[1]}

                    try:
                        next_grant = pred[0] - self.env.now#time until next grant begining
                        yield self.env.timeout(next_grant)#wait for the next grant
                    except Exception as e:
                        logging.debug("Error while waiting for the next grant ({})".format(e))
                        pass

                    self.port.set_grant(pred_grant)
                    sent_pkt = self.env.process(self.port.sent(self.oid))#sending predicted messages

                    yield sent_pkt
                    grant_usage = yield self.port.grant_real_usage.get()
                    if len(grant_usage) > 0:
                        pred_grant_usage_report.append(grant_usage)
                    else:
                        logging.debug("Erro in grant_usage")

            yield self.grant_report_store.put(pred_grant_usage_report)#Signals the end of grant processing

    def ONU_sender(self, cable):
        """A process which checks the queue size and send a REQUEST message to OLT"""
        while True:

            if self.port.byte_size >= self.bucket:# send a REQUEST only if the queue size is greater than the bucket size
                requested_buffer = self.port.byte_size #gets the size of the buffer that will be requested
                self.port.update_last_buffer_size(requested_buffer)#update the size of the current/last buffer REQUEST
                msg = {'text':"ONU %s sent this REQUEST for %.6f at %f" %
                    (self.oid,self.port.byte_size, self.env.now),'buffer_size':requested_buffer,'ONU':self}
                cable.put_request((msg),self.delay)# put the request message in the cable
                self.grant_report = yield self.grant_report_store.get()# Wait for the grant processing to send the next request
            else:
                yield self.env.timeout(self.delay)

class DBA(object):
    def __init__(self,env,max_grant_size,grant_store):
        self.env = env
        self.max_grant_size = max_grant_size
        self.grant_store = grant_store
        self.guard_interval = 0.000001

class DBA_IPACT(DBA):
    def __init__(self,env,max_grant_size,grant_store):
        DBA.__init__(self,env,max_grant_size,grant_store)
        self.counter = simpy.Resource(self.env, capacity=1)#create a queue of requests to DBA


    def ipact(self,ONU,buffer_size):
        with self.counter.request() as my_turn:
            yield my_turn
            time_stamp = self.env.now
            delay = ONU.delay

            if self.max_grant_size > 0 and buffer_size > self.max_grant_size:
                buffer_size = self.max_grant_size
            bits = buffer_size * 8
            sending_time = 	bits/float(1000000000)
            grant_time = delay + sending_time + self.guard_interval
            grant_final_time = self.env.now + grant_time
            counter = Grant_ONU_counter[ONU.oid]
            grant_time_file.write( "{},{},{},{},{},{},{},{}\n".format(MAC_TABLE['olt'], MAC_TABLE[ONU.oid],"02", time_stamp,counter, ONU.oid,self.env.now,grant_final_time) )
            grant = {'ONU':ONU,'grant_size': buffer_size, 'grant_final_time': grant_final_time, 'prediction': None}
            self.grant_store.put(grant)
            Grant_ONU_counter[ONU.oid] += 1

            yield self.env.timeout(grant_time)

class DBA_PRED_FILE(DBA):
    def __init__(self,env,max_grant_size,grant_store):
        DBA.__init__(self,env,max_grant_size,grant_store)
        self.counter = simpy.Resource(self.env, capacity=1)#create a queue of requests to DBA
        self.PREDICTION_FILE = {}
        for i in range(NUMBER_OF_ONUs):
            self.PREDICTION_FILE[i] = []
        self.predictor_file()

    def predictor_file(self):
        file_pred = open("grant.pred",'r')
        allpred = file_pred.read()
        file_pred.close()
        allpred = allpred.split()
        for pred in allpred:
            splitpred = pred.split(',')
            self.PREDICTION_FILE[int(splitpred[0])].append([float(splitpred[1]),float(splitpred[2])])

    def ipact_pred_file(self,ONU,buffer_size):
        with self.counter.request() as my_turn:
            yield my_turn

            delay = ONU.delay

            if self.max_grant_size > 0 and buffer_size > self.max_grant_size:
                buffer_size = self.max_grant_size
            bits = buffer_size * 8
            sending_time = 	bits/float(1000000000)
            grant_time = delay + sending_time + self.guard_interval
            grant_final_time = self.env.now + grant_time

            #PREDICTIONS
            if len(self.PREDICTION_FILE[ONU.oid]) > 0:
                prediction = self.PREDICTION_FILE[ONU.oid]
                self.PREDICTION_FILE[ONU.oid] = []
            else:
                prediction = None

            #grant_time_file.write( "{},{},{}\n".format(ONU.oid,self.env.now,grant_final_time) )
            grant = {'ONU':ONU,'grant_size': buffer_size, 'grant_final_time': grant_final_time, 'prediction': prediction}
            self.grant_store.put(grant)

            yield self.env.timeout(grant_time)


class PD_DBA(DBA):
    def __init__(self,env,max_grant_size,grant_store,window=20,predict=5):
        DBA.__init__(self,env,max_grant_size,grant_store)
        self.counter = simpy.Resource(self.env, capacity=1)#create a queue of requests to DBA
        self.window = window
        self.predict = predict
        self.predictions_array = []
        self.grant_history = range(NUMBER_OF_ONUs)
        for i in range(NUMBER_OF_ONUs):
            self.grant_history[i] = {'counter': [], 'start': [], 'end': []}

    def predictions_schedule(self,predictions):
        if len(self.predictions_array) > 0:
            self.predictions_array = filter(lambda x: x[0] > self.env.now, self.predictions_array)
        self.predictions_array +=  predictions
        self.predictions_array.sort()
        j = 1
        for interval1 in self.predictions_array[:-1]:
            for interval2 in self.predictions_array[j:]:
                if interval1[1] > interval2[0]:
                    if interval1 in predictions:
                        index1 = self.predictions_array.index(interval1)
                        index2 = predictions.index(interval1)
                        new_interval = [ interval1[0] , interval2[0] - self.guard_interval ]
                        predictions[ index2 ] = new_interval
                        self.predictions_array[index1] = new_interval

                    else:
                        index1 = self.predictions_array.index(interval2)
                        index2 = predictions.index(interval2)
                        new_interval = [ interval1[1] + self.guard_interval, interval2[1] ]
                        predictions[ index2 ] = new_interval
                        self.predictions_array[index1] = new_interval


                else:
                    break
            j+=1
        return predictions


    def predictor_online(self, ONU_id):
        if len( self.grant_history[ONU_id]['start'] ) >= self.window :
            df_tmp = pd.DataFrame(self.grant_history)
            X_pred = np.arange(self.grant_history[ONU_id]['counter'][-1] +1, self.grant_history[ONU_id]['counter'][-1] + 1 + self.predict).reshape(-1,1)
            #predicting start time
            reg = linear_model.LinearRegression()
            reg.fit( np.array( df_tmp['counter'] ).reshape(-1,1) , df_tmp['start'] )
            start_pred = reg.predict(X_pred)
            #predicting end time
            reg = linear_model.LinearRegression()
            reg.fit( np.array( df_tmp['counter'] ).reshape(-1,1) , df_tmp['end'] )
            end_pred = reg.predict(X_pred)

            #merging start_pred and end_pred
            predictions = []
            for i in range(len(start_pred)):
                predictions.append( [ start_pred[i], end_pred[i] ] )

            #fixing overlap
            predictions = predictions_schedule(predictions)
            return predictions
        else:
            return None



    def pd_dba(self,ONU,buffer_size,predictions_report):
        with self.counter.request() as my_turn:
            yield my_turn
            time_stamp = self.env.now
            delay = ONU.delay
            if predictions_report:
                self.update_prediction_report(predictions_report)

            if self.max_grant_size > 0 and buffer_size > self.max_grant_size:
                buffer_size = self.max_grant_size
            bits = buffer_size * 8
            sending_time = 	bits/float(1000000000)
            grant_time = delay + sending_time + self.guard_interval
            grant_final_time = self.env.now + grant_time

            self.grant_history[ONU.id]['start'].append(self.env.now)
            self.grant_history[ONU.id]['end'].append(grant_final_time)
            self.grant_history[ONU.id]['counter'].append( len( self.grant_history[ONU.id]['start'] ) )

            #PREDICTIONS
            prediction = predictor_online(ONU.id)

            #grant_time_file.write( "{},{},{}\n".format(ONU.oid,self.env.now,grant_final_time) )
            grant = {'ONU':ONU,'grant_size': buffer_size, 'grant_final_time': grant_final_time, 'prediction': prediction}
            self.grant_store.put(grant)

            yield self.env.timeout(grant_time)




class OLT(object):
    def __init__(self,env,cable,max_grant_size):
        self.env = env
        self.grant_store = simpy.Store(self.env)
        self.dba = DBA_IPACT(self.env, max_grant_size, self.grant_store)
        self.receiver = self.env.process(self.OLT_receiver(cable))#
        self.sender = self.env.process(self.OLT_sender(cable))#

    def OLT_sender(self,cable):
        """A process which sends a grant message to ONU"""
        while True:
            grant = yield self.grant_store.get()
            cable.put_grant(grant['ONU'],grant)

    def OLT_receiver(self,cable):
        """A process which receives a request message from the ONUs."""
        while True:
            request = yield cable.get_request()#get a request message
            print("Received Request from ONU {} at {}".format(request['ONU'].oid, self.env.now))
            self.env.process(self.dba.ipact(request['ONU'],request['buffer_size']))


#starts the simulator
random.seed(RANDOM_SEED)
env = simpy.Environment()
cable = Cable(env)
ONU_List = []

for i in range(NUMBER_OF_ONUs):
    MAC_TABLE[i] = "00:00:00:00:{}:{}".format(random.randint(0x00, 0xff),random.randint(0x00, 0xff))
    Grant_ONU_counter[i] = 0
MAC_TABLE['olt'] = "ff:ff:ff:ff:00:01"
for i in range(NUMBER_OF_ONUs):
    #distance = random.randint(19,DISTANCE)
    distance= DISTANCE
    exp=116*10#arbitrary value for the exponential distribution
    ONU_List.append(ONU(distance,i,env,cable,exp,ONU_QUEUE_LIMIT,PKT_SIZE,MAX_BUCKET_SIZE))

olt = OLT(env,cable,MAX_GRANT_SIZE)
logging.info("starting simulator")
env.run(until=SIM_DURATION)
delay_file.close()
grant_time_file.close()
pkt_file.close()

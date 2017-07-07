import simpy
import random
import functools
import time
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse


#Parsing the inputs arguments
parser = argparse.ArgumentParser(description="Long Reach PON Simulator")
group = parser.add_mutually_exclusive_group()
#group.add_argument("-v", "--verbose", action="store_true")
group.add_argument("-q", "--quiet", action="store_true")
parser.add_argument("n", type=int, help="the number of ONUs")
parser.add_argument("b", type=int, default=0, help="the size of the ONU sender bucket in bytes")
parser.add_argument("q", type=int, default=0, help="the size of the ONU port queue in bytes")
args = parser.parse_args()

#Arguments
NUMBER_OF_ONUs= args.n

#settings
SIM_DURATION = 30
RANDOM_SEED = 20
PKT_SIZE = 9000


class Cable(object):
    """This class represents the propagation through a cable and the splitter."""
    def __init__(self, env, delay):
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
    def __init__(self, env, id,  adist, sdist, initial_delay=0, finish=float("inf"), flow_id=0, fix_pkt_size=None):
        self.id = id
        self.env = env
        self.arrivals_dist = adist #packet arrivals distribution
        self.size_dist = sdist #packet size distribution
        self.initial_delay = initial_delay
        self.fix_pkt_size = fix_pkt_size
        self.finish = finish
        self.out = None
        self.packets_sent = 0
        self.action = env.process(self.run())  # starts the run() method as a SimPy process
        self.flow_id = flow_id


    def run(self):
        """The generator function used in simulations.
        """

        yield self.env.timeout(self.initial_delay)
        while self.env.now < self.finish:
            # wait for next transmission
            yield self.env.timeout(self.arrivals_dist())
            self.packets_sent += 1

            if self.fix_pkt_size:
                p = Packet(self.env.now, self.fix_pkt_size, self.packets_sent, src=self.id, flow_id=self.flow_id)
            else:
                p = Packet(self.env.now, self.size_dist(), self.packets_sent, src=self.id, flow_id=self.flow_id)

            self.out.put(p)

class ONUPort(object):


    def __init__(self, env, rate, qlimit=None):
        self.store = simpy.Store(env)#buffer
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

    def update_last_buffer_size(self,last_buffe_size): #update the size of the last buffer request
        self.last_buffer_size = self.byte_size

    def get_last_buffer_size(self): #return the size of the last buffer request
        return self.last_buffer_size

    def get_pkt(self):
        try:
            pkt = (yield self.store.get() )#getting a packet from the buffer
            self.pkt = pkt
        except simpy.Interrupt as i:

            pass

        if not self.grant_loop:#put the pkt back to the buffer if the grant time expired
            self.store.put(pkt)



    def sent(self,ONU_id):
        self.grant_loop = True
        while self.grant_final_time > self.env.now:

            get_pkt = self.env.process(self.get_pkt())#trying to get a package in the buffer
            grant_timeout = self.env.timeout(self.grant_final_time - self.env.now)
            yield get_pkt | grant_timeout#wait for a package to be sent or the grant timeout

            if (self.grant_final_time <= self.env.now):
                #The grant time has expired
                break
            if self.pkt is not None:
                pkt = self.pkt
            else:
                #there is not packate to be sent
                break
            self.busy = 1
            self.byte_size -= pkt.size
            if self.byte_size < 0:#Prevent the buffer from being negative

                self.byte_size += pkt.size
                self.store.put(pkt)
                break

            bits = pkt.size * 8
            sending_time = 	bits/float(1000000000)
            #To avoid fragmentation by passing the Grant window
            if env.now + sending_time > self.grant_final_time + self.guard_interval:
                self.byte_size += pkt.size

                self.store.put(pkt)
                break
            yield self.env.timeout(sending_time)

            self.pkt = None
        self.grant_loop = False #ending of the grant



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
            return self.store.put(pkt)
        if tmp >= self.qlimit: # chcks if the queue is full
            self.packets_drop += 1
            #return
        else:
            self.byte_size = tmp
            self.store.put(pkt)

class ONU(object):
    def __init__(self,distance,oid,env,cable,exp,qlimit,fix_pkt_size,bucket):
        self.env = env
        self.grant_store = simpy.Store(self.env) #Stores grant_size
        self.distance = distance #fiber distance
        self.oid = oid #ONU indentifier
        self.delay = self.distance/ float(210000) # fiber propagation delay
        self.thread_delay = 0 #remover
        self.excess = 0 #difference between the size of the request and the grant
        arrivals_dist = functools.partial(random.expovariate, exp) #packet arrival distribuition
        size_dist = functools.partial(random.expovariate, 0.01)  # packet size distribuition, mean size 100 bytes
        samp_dist = functools.partial(random.expovariate, 1.0)#remover
        port_rate = 1000.0 #para que serve?
        self.pg = PacketGenerator(self.env, "bbmp", adist, sdist,fix_pkt_size) #creates the packet generator
        if qlimit ==0:# checks if the queue has a size limit
            queue_limit = None
        else:
            queue_limit = qlimit
        self.port = ONUPort(self.env, port_rate, qlimit=queue_limit)#create ONU PORT
        self.pg.out = self.port #forward packet generator output to ONU port
        self.sender = self.env.process(self.ONU_sender(cable))
        self.receiver = self.env.process(self.ONU_receiver(cable))
        self.bucket = bucket #Bucket size

    def ONU_receiver(self,cable):
        while True:
            grant = yield cable.get_grant(self.oid)#waiting for a grant

            self.excess = self.port.get_last_buffer_size - grant['grant_size'] #update the excess
            self.port.set_grant(grant)

            sent_pkt = self.env.process(self.port.sent(self.oid))
            yield sent_pkt

            if grant['prediction']:#check if have any predicion in the grant
                for pred in grant_prediction[1:]:
                    pred_grant = {'grant_size': grant['grant_size'], 'grant_final_time': pred[1]}

                    try:
                        next_grant = pred[0] - self.env.now#time until next grant begining
                        yield self.env.timeout(next_grant)#wait for the next grant
                    except Exception as e:
                        #print e
                        pass

                    self.port.set_grant(pred_grant)
                    sent_pkt = self.env.process(self.port.sent(self.oid))#sending predicted messages

                    yield sent_pkt


            yield self.grant_store.put("ok")#Signals the end of grant processing

    def ONU_sender(self, cable):
        """A process which checks the queue size and send a REQUEST message to OLT"""
        while True:

            if self.port.byte_size >= self.bucket:# send a REQUEST only if the queue size is greater than the bucket size

                self.port.update_last_buffer_size()#update the size of the current/last buffer REQUEST
                msg = {'text':"ONU %s sent this REQUEST for %.6f at %f" %
                    (self.oid,self.port.byte_size, self.env.now),'buffer_size':self.port.byte_size,'ONU':self}
                cable.put((msg),self.delay)# put the request message in the cable
                yield self.grant_store.get()# Wait for the grant processing to send the next request
            else:
                yield self.env.timeout(self.delay)

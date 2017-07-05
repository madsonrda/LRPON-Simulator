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
        self.store = simpy.Store(env)
        self.res = simpy.Resource(env, capacity=1)
        self.grant_size = 0
        self.grant_final_time = 0
        self.guard_int = 0.000001
        self.rate = rate #remover
        self.env = env
        self.out = None
        self.packets_rec = 0
        self.packets_drop = 0
        self.qlimit = qlimit
        self.byte_size = 0  # Current size of the queue in bytes
        self.last_byte_size = 0
        self.debug = debug
        self.busy = 0  # Used to track if a packet is currently being sent
        self.action = env.process(self.run())  # starts the run() method as a SimPy process
        self.msg = None
        self.grant_loop = False
        self.usage = 0 #para que serve?

    def get_usage(self): #para que serve?
        return self.usage

    def reset_usage(self):#para que serve?
        self.usage = 0

    def set_grant(self,grant): #setting grant byte size and its ending
        self.grant_size = grant['grant_size']
        self.grant_final_time = grant['grant_final_time']

    def set_last_b_size(self,last_b_size): #setting the size of the queue in bytes
        self.last_byte_size = last_b_size

    def get_msg(self):
        try:
            msg = (yield self.store.get() )
            self.msg = msg
        except simpy.Interrupt as i:

            pass

        if not self.grant_loop:
            self.store.put(msg)



    def sent(self,ONU_id):
        self.grant_loop = True
        while self.grant_final_time > self.env.now:

            get_msg = self.env.process(self.get_msg())
            grant_timeout = self.env.timeout(self.grant_final_time - self.env.now)
            yield get_msg | grant_timeout
            #print self.env.now
            if (self.grant_final_time <= self.env.now):
                #print "acabou tempo da grant"
                #get_msg.interrupt('acabou tempo da grant')
                break
            if self.msg is not None:
                msg = self.msg
            else:
                #print "msg None"
                # if not get_msg.triggered:
                #     get_msg.interrupt('nao ha pacote no buffer')
                break
            self.busy = 1
            # if (self.grant_size - msg.size) < -1:
            #     self.store.put(msg)
            #     #print "nao enviei (pouco grant)"
            #     break
            #print ("msg_size = %s" % (msg.size))
            self.byte_size -= msg.size
            if self.byte_size < 0:
                #print("ONU %s: buffer negativo %s" % (ONU_id,self.byte_size))
                self.byte_size += msg.size
                self.store.put(msg)
                break
            #self.grant_size -= msg.size
            bits = msg.size * 8
            sending_time = 	bits/float(1000000000)
            #Prara evitar fragmentacao se passar o a janela do grant
            if env.now + sending_time > self.grant_final_time + self.guard_int:
                self.byte_size += msg.size
                #self.grant_size += msg.size
                self.store.put(msg)
                #print "nao enviei (pouco tempo)"
                break
            yield self.env.timeout(sending_time)
            self.usage += sending_time
            UTILIZATION.append(sending_time)
            Throughput.append(msg.size)
            #print("paket delay is %.6f" % (self.env.now - msg.time))
            Delay.append(self.env.now - msg.time)
            #print("ENVIEI % s" % (msg))
            self.msg = None
        self.grant_loop = False



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
        self.grant_wait = simpy.Store(self.env) #para que serve?
        self.distance = distance #fiber distance
        self.oid = oid #ONU indentifier
        self.delay = self.distance/ float(210000) # fiber propagation delay
        self.thread_delay = 0 #remover
        self.last_request_size = 0 #para que serve?
        self.excess = 0 #para que serve?
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
            grant = yield cable.get_grant(self.oid)
            grant_size = grant['grant_size']
            grant_final_time = grant['grant_final_time']
            grant_prediction = grant['prediction']
            self.excess = self.last_request_size - grant_size
            self.port.set_grant(grant)
            start = self.env.now
            SLOT.append(grant_final_time - start)
            sent_pkt = self.env.process(self.port.sent(self.oid))
            yield sent_pkt
            #print ((grant_final_time - start)-self.port.get_usage())/(grant_final_time - start)
            self.port.reset_usage()
            if grant_prediction:
                for pred in grant_prediction[1:]:
                    pred_grant = {'grant_size': grant_size, 'grant_final_time': pred[1]}
                    #print("ONU %s:%s,%s"% (self.oid,pred,self.env.now))
                    try:
                        next_grant = pred[0] - self.env.now
                        yield self.env.timeout(next_grant)
                    except Exception as e:
                        #print e
                        pass

                    self.port.set_grant(pred_grant)
                    sent_pkt = self.env.process(self.port.sent(self.oid))
                    start = self.env.now
                    yield sent_pkt
                    #print ((pred[1]-pred[0])- self.port.get_usage())/(pred[1]-pred[0])
                    self.port.reset_usage()
                    SLOT.append(pred[1]-start)
            yield self.grant_wait.put("ok")

    def ONU_sender(self, cable):
        """A process which checks the queue size and send a REQUEST message to OLT"""
        while True:

            if self.port.byte_size >= self.bucket:# send a REQUEST only if the queue size is greater than the bucket size
                self.last_request_size = self.port.byte_size #update the size of the current buffer REQUEST in the ONU class
                self.port.set_last_b_size(self.port.byte_size)#repetido?
                msg = {'text':"ONU %s sent this REQUEST for %.6f at %f" %
                    (self.oid,self.port.byte_size, self.env.now),'buffer_size':self.port.byte_size,'ONU':self}
                cable.put((msg),self.delay)# put the request message in the cable
                yield self.grant_wait.get()# Wait for the grant to send the next request
            else:
                yield self.env.timeout(self.delay)

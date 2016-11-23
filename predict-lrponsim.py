import simpy
import random
import functools
import time
import numpy
import sys
import ast

SIM_DURATION = 10000
RANDOM_SEED = 42
REQUEST = []
NUMBER_OF_ONUs = 3
PKT_SIZE = 9000
Delay = []
LIMITE = 12005
BUFFER = open('b1','rb')
ONU0PRED = open('onu0pred','rb')
ONU1PRED = open('onu1pred','rb')
ONU2PRED = open('onu2pred','rb')




class Cable(object):
    """This class represents the propagation through a cable."""
    def __init__(self, env, delay):
        self.env = env
        self.upstream = simpy.Store(env)
        self.downstream = []
        for i in range(NUMBER_OF_ONUs):
            self.downstream.append(simpy.Store(env))

    def up_latency(self, value,delay):
        yield self.env.timeout(delay)
        self.upstream.put(value)

    def down_latency(self,ONU,value):
        yield self.env.timeout(ONU.delay)
        self.downstream[ONU.oid].put(value)


    def put(self, value,delay):
        self.env.process(self.up_latency(value,delay))

    def get(self):
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
    def __init__(self, env,  adist, sdist, initial_delay=0, finish=float("inf"), flow_id=0):
        self.id = "bla"
        self.env = env
        self.adist = adist
        self.sdist = sdist
        self.initial_delay = initial_delay
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
            yield self.env.timeout(self.adist())
            self.packets_sent += 1
            # print "PKT_SENT"
            size = int(BUFFER.readline())
            #size = random.normalvariate(15,0.5)
            #p = Packet(self.env.now, self.sdist(), self.packets_sent, src=self.id, flow_id=self.flow_id)
            p = Packet(self.env.now, size, self.packets_sent, src=self.id, flow_id=self.flow_id)
            self.out.put(p)


class SwitchPort(object):


    def __init__(self,oid, env, rate, qlimit=None, debug=False):
        self.oid = oid
        self.store = simpy.Store(env)
        self.res = simpy.Resource(env, capacity=1)
        self.grant_size = 0
        self.rate = rate
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

    def set_grant(self,grant_size):
        self.grant_size = grant_size
    def set_last_b_size(self,last_b_size):
        self.last_byte_size = last_b_size

    def sent(self,ONU_id):
        print("Momento em que esta liberado para enviar os pacotes %s" % self.env.now)
        print("ONU %s: request %s, grant %s, current_buffer %s" % (ONU_id,self.last_byte_size,self.grant_size,self.byte_size))
        while self.grant_size > 0:
            msg = (yield self.store.get())
            print("Momento em que ha pacotes a serem enviados  %s" % self.env.now)
            self.busy = 1
            # print("%f" % self.grant_size)
            # print("%f" % msg.size)
            # print self.grant_size - msg.size
            if (self.grant_size - msg.size) < -1:
                self.store.put(msg)
                break
            self.byte_size -= msg.size
            self.grant_size -= msg.size
            bits = msg.size * 8
            sending_time = 	bits/float(1000000000)
            yield self.env.timeout(sending_time)
            #print("paket delay is %.6f" % (self.env.now - msg.time))
            Delay.append(self.env.now - msg.time)


    def run(self):
        while True:
            yield self.env.timeout(5)
                # msg = (yield self.store.get())
                # self.busy = 1
                # self.byte_size -= msg.size
                # self.grant_size -= msg.size
                # yield self.env.timeout(msg.size*8.0/self.rate)
                # #self.out.put(msg)
                # self.busy = 0
                # if self.debug:
                #     print msg

    def put(self, pkt):
        self.packets_rec += 1
        tmp = self.byte_size + pkt.size

        if self.qlimit is None:
            self.byte_size = tmp
            return self.store.put(pkt)
        if tmp >= self.qlimit:
            self.packets_drop += 1
            return
        else:
            self.byte_size = tmp
            self.store.put(pkt)
            #print("ONU: %s current buffer %s" % (self.oid,self.byte_size))

class ONU(object):
    def __init__(self,distance,oid,env,cable):
        self.env = env
        self.distance = distance
        self.oid = oid
        self.delay = self.distance/ float(210000)
        self.thread_delay = 0
        self.last_req = 0
        self.excess = 0
        adist = functools.partial(random.expovariate, 0.5)
        sdist = functools.partial(random.expovariate, 0.01)  # mean size 100 bytes
        samp_dist = functools.partial(random.expovariate, 1.0)
        port_rate = 1000.0
        self.pg = PacketGenerator(self.env, adist, sdist)
        self.port = SwitchPort(self.oid,self.env, port_rate, qlimit=1000000)
        self.pg.out = self.port
        self.sender = self.env.process(self.ONU_sender(cable))

    def ONU_sender(self, cable):
        """A process which randomly generates messages."""
        while True:
            #b_size = self.port.byte_size
            #print "######################"
            if  self.port.byte_size > 0:
                self.last_req = self.port.byte_size
                self.port.set_last_b_size(self.port.byte_size)
                msg = {'text':"ONU %s sent this REQUEST for %.6f at %f" %
                    (self.oid,self.port.byte_size, self.env.now),'buffer_size': self.port.byte_size,'ONU':self}
                cable.put((msg),self.delay)

                grant = yield cable.get_grant(self.oid)
                for grant_size in grant['prediction']:
                    self.excess = self.last_req - grant_size
                #b_size =self.port.byte_size
                #self.newly_arrived.append(b_size - self.last_req)
                    self.port.set_grant(grant_size)
                    sent_pkt = self.env.process(self.port.sent(self.oid))
                    yield sent_pkt
            else:
                yield self.env.timeout(self.delay)

class OLT(object):
    def __init__(self,env,cable):
        self.env = env
        self.guard_int = 0.000001
        self.counter = simpy.Resource(self.env, capacity=1)
        self.receiver = self.env.process(self.OLT_receiver(cable))

    def DBA_IPACT(self,ONU,b_size,cable):
        with self.counter.request() as my_turn:
            yield my_turn
            delay = ONU.delay
            bits = b_size * 8
            sending_time = 	bits/float(1000000000)
            grant_time = delay + sending_time + self.guard_int
            #print("ONU %s: grant time for %s is %f" % (ONU.oid,b_size,grant_time))
            #enviar pelo cabo o buffer para a onu
            msg = {'grant_size': b_size}
            cable.put_grant(ONU,msg)
            yield self.env.timeout(grant_time)

    def predictor(self,ONU_id):
        print ONU_id
        if ONU_id == 0:
            print "pred 0"
            return ast.literal_eval(ONU0PRED.readline())
        if ONU_id == 1:
            print "pred 1"
            return ast.literal_eval(ONU1PRED.readline())
        if ONU_id == 2:
            print "pred 2"
            return ast.literal_eval(ONU2PRED.readline())

    def DBA_PRED(self,ONU,b_size,cable):
        with self.counter.request() as my_turn:
            yield my_turn
            prediction = self.predictor(ONU.oid)
            print prediction
            sending_time = 0
            delay = ONU.delay
            for grant in prediction:
                bits = grant * 8
                sending_time = 	bits/float(1000000000)
            grant_time = delay + sending_time + self.guard_int
                #print("ONU %s: grant time for %s is %f" % (ONU.oid,b_size,grant_time))
                #enviar pelo cabo o buffer para a onu
            msg = {'grant_size': b_size, 'prediction': prediction}
            cable.put_grant(ONU,msg)
            yield self.env.timeout(grant_time)
            #return grant_time

    def OLT_receiver(self,cable):
        """A process which consumes messages."""
        while True:
            # Get event for message pipeONU0PRED = open('onu0pred','rb')
            #pkt_size = random.randint(5,20)
            msg = yield cable.get()
            #print('OLT Received this at %f while %s' % (self.env.now, msg['text']))
            REQUEST.append((msg['ONU'].oid,msg['buffer_size'],self.env.now))
            print("%s,%s,%s,%f" % (msg['ONU'].oid,msg['buffer_size'],msg['ONU'].excess,self.env.now))
            self.env.process(self.DBA_IPACT(msg['ONU'],msg['buffer_size'],cable))
            #dba = self.DBA_IPACT(msg['ONU'].delay,msg['queue_size'])
            #yield self.env.timeout(dba)


# Setup and start the simulation
#print('Event Latency')
print("ONU,buffer,excess,time")
random.seed(RANDOM_SEED)
env = simpy.Environment()


cable = Cable(env, 10)
ONU_List = []
for i in range(NUMBER_OF_ONUs):
    distance = random.randint(60,100)
    ONU_List.append(ONU(distance,i,env,cable))

olt = OLT(env,cable)
try:
    env.run(until=SIM_DURATION)
except Exception as e:
    pass

BUFFER.close()
#print REQUEST[-6:]
print numpy.mean(Delay)

import simpy
import random
import functools
import time
import numpy
import pandas as pd
import matplotlib.pyplot as plt

SIM_DURATION = 30
RANDOM_SEED = 20
REQUEST = []
NUMBER_OF_ONUs = 30
PKT_SIZE = 9000

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
    def __init__(self, env, id,  adist, sdist, initial_delay=0, finish=float("inf"), flow_id=0):
        self.id = id
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
            size = PKT_SIZE
            PKT.append(size)
            #p = Packet(self.env.now, self.sdist(), self.packets_sent, src=self.id, flow_id=self.flow_id)
            p = Packet(self.env.now, size, self.packets_sent, src=self.id, flow_id=self.flow_id)
            self.out.put(p)


class SwitchPort(object):


    def __init__(self, env, rate, qlimit=None, debug=False):
        self.store = simpy.Store(env)
        self.res = simpy.Resource(env, capacity=1)
        self.grant_size = 0
        self.grant_final_time = 0
        self.guard_int = 0.000001
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
        self.msg = None
        self.grant_loop = False

    def set_grant(self,grant):
        self.grant_size = grant['grant_size']
        self.grant_final_time = grant['grant_final_time']

    def set_last_b_size(self,last_b_size):
        self.last_byte_size = last_b_size

    def get_msg(self):
        try:
            msg = (yield self.store.get() )
            #print("Momento em que ha pacotes a serem enviados  %s" % self.env.now)
            #print msg
            self.msg = msg
        except simpy.Interrupt as i:
            print('get_msg interrupted at', env.now, 'msg:', i.cause)
        #print self.grant_loop
        if not self.grant_loop:
            self.store.put(msg)
            #print "pacote perdido"


    # def sent(self,ONU_id):
    #     self.grant_loop = True
    #     #print("ONU %s: buffer-req %s, grant %s, current_buffer %s" % (ONU_id,self.last_byte_size,self.grant_size,self.byte_size))
    #     #print("Momento em que esta liberado para enviar os pacotes %s" % self.env.now)
    #     #print self.env.now
    #     #print (self.grant_size * 8)/float(1000000000)
    #     #grant_final_time = env.now + (self.grant_size * 8)/float(1000000000)
    #     print ("ONU %s: tempo limite do grant %s" % (ONU_id, self.grant_final_time))
    #     while self.grant_size > 0:
    #         #print ("grant_size = %s" % (self.grant_size))
    #         #print self.env.now
    #         #msg = (yield self.store.get() | self.env.timeout(grant_final_time - self.env.now)).values()
    #         get_msg = self.env.process(self.get_msg())
    #         grant_timeout = self.env.timeout(self.grant_final_time - self.env.now)
    #         #print self.env.now
    #         yield get_msg | grant_timeout
    #         #print self.env.now
    #         #print grant_final_time
    #         if (self.grant_final_time < env.now):
    #             print "acabou tempo da grant"
    #             break
    #         if self.msg is not None:
    #             msg = self.msg
    #         else:
    #             #print "msg None"
    #             break
    #         self.busy = 1
    #         if (self.grant_size - msg.size) < -1:
    #             self.store.put(msg)
    #             #print "nao enviei (pouco grant)"
    #             break
    #         #print ("msg_size = %s" % (msg.size))
    #         self.byte_size -= msg.size
    #         self.grant_size -= msg.size
    #         bits = msg.size * 8
    #         sending_time = 	bits/float(1000000000)
    #         #Prara evitar fragmentacao se passar o a janela do grant
    #         if env.now + sending_time > self.grant_final_time + self.guard_int:
    #             self.byte_size += msg.size
    #             self.grant_size += msg.size
    #             self.store.put(msg)
    #             #print env.now + sending_time
    #             #print "nao enviei (pouco tempo)"
    #             break
    #         yield self.env.timeout(sending_time)
    #         #print("paket delay is %.6f" % (self.env.now - msg.time))
    #         Delay.append(self.env.now - msg.time)
    #         print("ENVIEI % s" % (msg))
    #         self.msg = None



    def sent(self,ONU_id):
        self.grant_loop = True
        #print ("ONU %s: tempo limite do grant %s" % (ONU_id, self.grant_final_time))
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
            UTILIZATION.append(sending_time)
            #print("paket delay is %.6f" % (self.env.now - msg.time))
            Delay.append(self.env.now - msg.time)
            #print("ENVIEI % s" % (msg))
            self.msg = None
        self.grant_loop = False



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
            print "drop"
            self.packets_drop += 1
            return
        else:
            self.byte_size = tmp
            self.store.put(pkt)

class ONU(object):
    def __init__(self,distance,oid,env,cable,exp):
        self.env = env
        self.grant_wait = simpy.Store(self.env)
        self.distance = distance
        self.oid = oid
        self.delay = self.distance/ float(210000)
        self.thread_delay = 0
        self.last_req = 0
        self.excess = 0
        adist = functools.partial(random.expovariate, exp)
        sdist = functools.partial(random.expovariate, 0.01)  # mean size 100 bytes
        samp_dist = functools.partial(random.expovariate, 1.0)
        port_rate = 1000.0
        self.pg = PacketGenerator(self.env, "Greg", adist, sdist)
        self.port = SwitchPort(self.env, port_rate, qlimit=None)
        self.pg.out = self.port
        self.sender = self.env.process(self.ONU_sender(cable))
        self.receiver = self.env.process(self.ONU_receiver(cable))

    def ONU_receiver(self,cable):
        while True:
            grant = yield cable.get_grant(self.oid)
            grant_size = grant['grant_size']
            grant_final_time = grant['grant_final_time']
            grant_prediction = grant['prediction']
            self.excess = self.last_req - grant_size
            self.port.set_grant(grant)
            sent_pkt = self.env.process(self.port.sent(self.oid))
            yield sent_pkt
            if grant_prediction:
                for pred in grant_prediction[1:]:
                    pred_grant = {'grant_size': grant_size, 'grant_final_time': pred[1]}
                    print("ONU %s:%s,%s"% (self.oid,pred,self.env.now))
                    next_grant = pred[0] - self.env.now
                    yield self.env.timeout(next_grant)
                    self.port.set_grant(pred_grant)
                    sent_pkt = self.env.process(self.port.sent(self.oid))
                    yield sent_pkt
            yield self.grant_wait.put("ok")

    def ONU_sender(self, cable):
        """A process which randomly generates messages."""
        while True:

            #b_size =self.port.byte_size

            if self.port.byte_size >= 18000:
                self.last_req = self.port.byte_size
                self.port.set_last_b_size(self.port.byte_size)
                msg = {'text':"ONU %s sent this REQUEST for %.6f at %f" %
                    (self.oid,self.port.byte_size, self.env.now),'buffer_size':self.port.byte_size,'ONU':self}
                cable.put((msg),self.delay)
                yield self.grant_wait.get()
            else:
                yield self.env.timeout(self.delay)

class OLT(object):
    def __init__(self,env,cable):
        self.env = env
        self.guard_int = 0.000001
        self.counter = simpy.Resource(self.env, capacity=1)
        self.receiver = self.env.process(self.OLT_receiver(cable))


    def predictor(self,ONU_id):
        #print ONU_id
        if ONU_id == 0:
            #print "pred 0"
            #return ast.literal_eval(ONU0PRED.readline())
            pred = ONU0PRED.read()
            p = pred.split()
            l = []
            for i in range(len(p)/2):
                l.append([float(p[2*i]),float(p[2*i+1])])
            return l
        if ONU_id == 1:
            #print "pred 1"
            #return ast.literal_eval(ONU1PRED.readline())
            pred = ONU1PRED.read()
            p = pred.split()
            l = []
            for i in range(len(p)/2):
                l.append([float(p[2*i]),float(p[2*i+1])])
            return l
        if ONU_id == 2:
            #print "pred 2"
            #return ast.literal_eval(ONU2PRED.readline())
            pred = ONU2PRED.read()
            p = pred.split()
            l = []
            for i in range(len(p)/2):
                l.append([float(p[2*i]),float(p[2*i+1])])
            return l

    def DBA_IPACT(self,ONU,b_size,cable):
        with self.counter.request() as my_turn:
            yield my_turn
            # try:
            #     prediction = self.predictor(ONU.oid)
            # except Exception as e:
            #     prediction = None
            delay = ONU.delay
            bits = b_size * 8
            sending_time = 	bits/float(1000000000)
            grant_time = delay + sending_time + self.guard_int
            grant_final_time = self.env.now +grant_time
            #print("ONU %s: grant time for %s is between %s and %s" %
                #(ONU.oid,b_size,self.env.now, grant_final_time))
            #enviar pelo cabo o buffer para a onu
            #print("%s,%s,%s" % (ONU.oid,self.env.now, grant_final_time))
            if not prediction_on:
                prediction = None
                PREDICTIONS[ONU.oid].append([self.env.now, grant_final_time])
            elif len(PREDICTIONS[ONU.oid]) > 0:
                prediction = PREDICTIONS[ONU.oid]
                PREDICTIONS[ONU.oid] = []
            else:
                prediction = None

            msg = {'grant_size': b_size, 'grant_final_time': grant_final_time, 'prediction': prediction}
            cable.put_grant(ONU,msg)
            yield self.env.timeout(grant_time)
            #return grant_time
    def OLT_receiver(self,cable):
        """A process which consumes messages."""
        while True:
            # Get event for message pipe
            #pkt_size = random.randint(5,20)
            msg = yield cable.get()
            #print('OLT Received this at %f while %s' % (self.env.now, msg['text']))
            REQUEST.append((msg['ONU'].oid,msg['buffer_size'],self.env.now))
            #print("%s,%s,%f" % (msg['ONU'].oid,msg['buffer_size'],self.env.now))
            self.env.process(self.DBA_IPACT(msg['ONU'],msg['buffer_size'],cable))
            #dba = self.DBA_IPACT(msg['ONU'].delay,msg['queue_size'])
            #yield self.env.timeout(dba)


# Setup and start the simulation
#print('Event Latency')
#print("ONU,buffer,time")
EXP = 116
DELAY = {}
LOAD = {}
PRED_DELAY = {}
PRED_LOAD = {}
PREDICTIONS = {}
for j in range(14):
    DELAY[j] = []
    LOAD[j]= []
    PRED_DELAY[j] = []
    PRED_LOAD[j]= []
for j in range(4):

    for k,exp in enumerate([87,116,145,174,203,232,261,
        290,319,348,377,406,435,464]):
    #for k,exp in enumerate([116]):
        prediction_on=False
        print "proximo sem pred"
        for i in range(NUMBER_OF_ONUs):
            PREDICTIONS[i] = []
        for m in range(2):
            UTILIZATION = []
            PKT = []
            Delay = []
            random.seed(RANDOM_SEED)
            env = simpy.Environment()


            cable = Cable(env, 10)
            ONU_List = []
            for i in range(NUMBER_OF_ONUs):
                #distance = random.randint(60,100)
                distance = 100
                ONU_List.append(ONU(distance,i,env,cable,exp))

            olt = OLT(env,cable)
            env.run(until=SIM_DURATION)
            if not prediction_on:
                DELAY[k].append(numpy.mean(Delay))
                LOAD[k].append(100*(numpy.sum(PKT)*8)/(float(SIM_DURATION)*float(1000000000)))
            else:
                PRED_DELAY[k].append(numpy.mean(Delay))
                PRED_LOAD[k].append(100*(numpy.sum(PKT)*8)/(float(SIM_DURATION)*float(1000000000)))
            prediction_on=True
            print "proximo com pred"
    RANDOM_SEED += 10

df = pd.DataFrame(DELAY)
df2 = pd.DataFrame(LOAD)
X=df2.mean().apply(numpy.round).values
Y=df.mean().values
delay_std = df.std().values

df3 = pd.DataFrame(PRED_DELAY)
df4 = pd.DataFrame(PRED_LOAD)
X1=df4.mean().apply(numpy.round).values
Y1=df3.mean().values
PRED_delay_std = df3.std().values
plt.figure()
plt.title("fig")
plt.xlabel("load")
plt.ylabel("delay")
plt.fill_between(X, Y - delay_std,Y + delay_std, alpha=0.1,color="r")
plt.fill_between(X1, Y1 - PRED_delay_std,Y1 + PRED_delay_std, alpha=0.1,color="b")
plt.plot(X, Y, 'o-', color="r",label="IPACT")
plt.plot(X1, Y1, 'o-', color="b",label="IPACT_PRED")
plt.show()
#print REQUEST[-6:]
# print("average pkt delay: %s" % (numpy.mean(Delay)))
# print("channel idle: %s%%" % (100-(100 * numpy.sum(UTILIZATION)/float(SIM_DURATION))))
# rate = (numpy.sum(PKT)*8)/(float(SIM_DURATION)*float(1000000000))
# print("upstream utilization: %s%%" % (100*(rate)))

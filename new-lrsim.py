import simpy
import random
import functools
import time
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import sys

SIM_DURATION = 30
RANDOM_SEED = 20
REQUEST = []
NUMBER_OF_ONUs = int(sys.argv[1])
PKT_SIZE = 9000

# ONU0PRED = open('onu0pred','rb')
# ONU1PRED = open('onu1pred','rb')
# ONU2PRED = open('onu2pred','rb')



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
        self.usage = 0

    def get_usage(self):
        return self.usage

    def reset_usage(self):
        self.usage = 0

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
            #print('get_msg interrupted at', env.now, 'msg:', i.cause)
            pass
        #print self.grant_loop
        if not self.grant_loop:
            self.store.put(msg)
            #print "pacote perdido"


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
            self.usage += sending_time
            UTILIZATION.append(sending_time)
            Throughput.append(msg.size)
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
            #print "drop"
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
        self.port = SwitchPort(self.env, port_rate, qlimit=sys.argv[6])
        self.pg.out = self.port
        self.sender = self.env.process(self.ONU_sender(cable))
        self.receiver = self.env.process(self.ONU_receiver(cable))
        self.bucket = float(sys.argv[4])

    def ONU_receiver(self,cable):
        while True:
            grant = yield cable.get_grant(self.oid)
            grant_size = grant['grant_size']
            grant_final_time = grant['grant_final_time']
            grant_prediction = grant['prediction']
            self.excess = self.last_req - grant_size
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
        """A process which randomly generates messages."""
        while True:

            #b_size =self.port.byte_size

            if self.port.byte_size >= self.bucket:
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



    def DBA_IPACT(self,ONU,b_size,cable):
        with self.counter.request() as my_turn:
            yield my_turn

            delay = ONU.delay
            b_max = float(sys.argv[5])
            if b_max > 0 and b_size > b_max:
                b_size = b_max
            bits = b_size * 8
            sending_time = 	bits/float(1000000000)
            grant_time = delay + sending_time + self.guard_int
            grant_final_time = self.env.now +grant_time
            #print("ONU %s: grant time for %s is between %s and %s" %
                #(ONU.oid,b_size,self.env.now, grant_final_time))
            #enviar pelo cabo o buffer para a onu
            #print("%s,%s,%s" % (ONU.oid,self.env.now, grant_final_time))
            if not prediction_on:
                #print "IPACT %s" % self.env.now
                prediction = None
                PREDICTIONS[ONU.oid].append([self.env.now, grant_final_time])
            elif prediction_file and len(PREDICTIONS_R[ONU.oid]) > 0:
                #print "proposta %s" % self.env.now
                prediction = PREDICTIONS_R[ONU.oid]
                PREDICTIONS_R[ONU.oid] = []
            elif AUX and len(PREDICTIONS[ONU.oid]) > 0:
                #print "perfeita %s" % self.env.now
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
EXP = [87,116,145,174,203,232,261,
    290,319,348,377,406,435,464]
#EXP = [406,435,464]
TIMES_SEED = int(sys.argv[2])
DELAY = {}
LOAD = {}
USAGE = {}
THROUGHPUT = {}
PRED_THROUGHPUT ={}
PRED_THROUGHPUT_R ={}
PRED_USAGE = {}
PRED_USAGE_R = {}
PRED_DELAY = {}
PRED_LOAD = {}
PRED_DELAY_R = {}
PRED_LOAD_R = {}
PREDICTIONS = {}
PREDICTIONS_R = {}
prediction_file = True
for j in range(len(EXP)):
    DELAY[j] = []
    LOAD[j]= []
    THROUGHPUT[j]=[]
    PRED_DELAY[j] = []
    PRED_LOAD[j]= []
    PRED_DELAY_R[j] = []
    PRED_LOAD_R[j]= []
    USAGE[j] = []
    PRED_USAGE[j] = []
    PRED_USAGE_R[j] = []
    PRED_THROUGHPUT[j] = []
    PRED_THROUGHPUT_R[j] =[]
for j in range(TIMES_SEED):

    for k,exp in enumerate(EXP):
    #for k,exp in enumerate([116]):
        prediction_on=False
        prediction_file = False
        AUX = True
        #print "proximo sem pred"
        for i in range(NUMBER_OF_ONUs):
            PREDICTIONS[i] = []
            PREDICTIONS_R[i] = []
        for m in range(2):
            UTILIZATION = []
            SLOT = []
            PKT = []
            Delay = []
            Throughput = []
            random.seed(RANDOM_SEED)
            env = simpy.Environment()


            cable = Cable(env, 10)
            ONU_List = []
            for i in range(NUMBER_OF_ONUs):
                #distance = random.randint(20,100)
                distance = int(sys.argv[3])
                ONU_List.append(ONU(distance,i,env,cable,exp))

            olt = OLT(env,cable)
            env.run(until=SIM_DURATION)
            if not prediction_on:
                DELAY[k].append(numpy.mean(Delay))
                load = 100*(numpy.sum(PKT)*8)/(float(SIM_DURATION)*float(1000000000))
                through = 100*(numpy.sum(Throughput)*8)/(float(SIM_DURATION)*float(1000000000))
                THROUGHPUT[k].append(through)
                #print "salvando resultado IPCAT load %d" % load
                #print ((numpy.sum(SLOT)-numpy.sum(UTILIZATION))/(numpy.sum(SLOT)))*100
                usage = (numpy.sum(UTILIZATION/numpy.sum(SLOT)))*100
                USAGE[k].append(usage)
                #time.sleep(8)
                LOAD[k].append(load)
                print("LOAD %d%%" % (load))
                print("ONU,inicio,fim")
                for ID in PREDICTIONS:
                    for tempos in PREDICTIONS[ID]:
                        print("%s,%s,%s" % (ID,tempos[0],tempos[1]))

                if prediction_file:
                    #print "lendo arquivo proposta"
                    time.sleep(1)
                    file_pred = open('result/load{0}.pred'.format(int(load)),'rb')
                    allpred = file_pred.read()
                    file_pred.close()
                    allpred = allpred.split()
                    for pred in allpred:
                        splitpred = pred.split(',')
                        PREDICTIONS_R[int(splitpred[0])].append([float(splitpred[1]),float(splitpred[2])])

            elif prediction_file:
                load = 100*(numpy.sum(PKT)*8)/(float(SIM_DURATION)*float(1000000000))
                #print "salvando resultado proposta load %d" % load
                # print ((numpy.sum(SLOT)-numpy.sum(UTILIZATION))/(numpy.sum(SLOT)))*100
                # print (numpy.sum(UTILIZATION/numpy.sum(SLOT)))*100
                usage = (numpy.sum(UTILIZATION/numpy.sum(SLOT)))*100
                PRED_USAGE_R[k].append(usage)
                through = 100*(numpy.sum(Throughput)*8)/(float(SIM_DURATION)*float(1000000000))
                PRED_THROUGHPUT_R[k].append(through)
                #time.sleep(5)
                PRED_DELAY_R[k].append(numpy.mean(Delay))
                PRED_LOAD_R[k].append(load)
                prediction_file = False
                AUX = True
            else:
                load = 100*(numpy.sum(PKT)*8)/(float(SIM_DURATION)*float(1000000000))
                #print "salvando resultado perfeita load %d" % load
                # print ((numpy.sum(SLOT)-numpy.sum(UTILIZATION))/(numpy.sum(SLOT)))*100
                # print (numpy.sum(UTILIZATION/numpy.sum(SLOT)))*100
                usage = (numpy.sum(UTILIZATION/numpy.sum(SLOT)))*100
                PRED_USAGE[k].append(usage)
                through = 100*(numpy.sum(Throughput)*8)/(float(SIM_DURATION)*float(1000000000))
                PRED_THROUGHPUT[k].append(through)
                #time.sleep(3)
                PRED_DELAY[k].append(numpy.mean(Delay))
                PRED_LOAD[k].append(load)
                AUX = False
            prediction_on=True
            #print "proximo com pred"
    RANDOM_SEED += 10

df_IPACT_delay = pd.DataFrame(DELAY)
df_IPACT_load = pd.DataFrame(LOAD)
df_IPACT_usage = pd.DataFrame(USAGE)
df_IPACT_throughput = pd.DataFrame(THROUGHPUT)
IPACT_LOAD=df_IPACT_load.mean().apply(numpy.round).values
IPACT_DELAY=df_IPACT_delay.mean().values
IPACT_delay_std = df_IPACT_delay.std().values
IPACT_USAGE =df_IPACT_usage.mean().values
IPACT_usage_std=df_IPACT_usage.std().values
IPACT_TRHOUGHPUT =df_IPACT_throughput.mean().values
IPACT_throughput_std=df_IPACT_throughput.std().values

print"#IPACT"
print "DELAY"
print df_IPACT_delay
print "mean %s" % IPACT_DELAY
print "std %s" % IPACT_delay_std
print "UTILIZATION"
print df_IPACT_usage
print "mean %s" % IPACT_USAGE
print "std %s" % IPACT_usage_std
print "THROUGHPUT"
print df_IPACT_throughput
print "mean %s" % IPACT_TRHOUGHPUT
print "std %s" % IPACT_throughput_std

df_PERF_DELAY = pd.DataFrame(PRED_DELAY)
df_PERF_LOAD = pd.DataFrame(PRED_LOAD)
df_PERF_USAGE = pd.DataFrame(PRED_USAGE)
df_PERF_throughput = pd.DataFrame(PRED_THROUGHPUT)
PERF_LOAD=df_PERF_LOAD.mean().apply(numpy.round).values
PERF_DELAY=df_PERF_DELAY.mean().values
PERF_delay_std = df_PERF_DELAY.std().values
PERF_USAGE = df_PERF_USAGE.mean().values
PERF_usage_std =df_PERF_USAGE.std().values
PERF_THROUGHPUT = df_PERF_throughput.mean().values
PERF_throughput_std =df_PERF_throughput.std().values

print"#PERFEITA"
print "DELAY"
print df_PERF_DELAY
print "mean %s" % PERF_DELAY
print "std %s" % PERF_delay_std
print "UTILIZATION"
print df_PERF_USAGE
print "mean %s" % PERF_USAGE
print "std %s" % PERF_usage_std
print "THROUGHPUT"
print df_PERF_throughput
print "mean %s" % df_PERF_throughput.mean().values
print "std %s" % PERF_throughput_std

# df_PROP_DELAY = pd.DataFrame(PRED_DELAY_R)
# df_PROP_LOAD = pd.DataFrame(PRED_LOAD_R)
# df_PROP_USAGE = pd.DataFrame(PRED_USAGE_R)
# df_PROP_throughput = pd.DataFrame(PRED_THROUGHPUT_R)
# PROP_LOAD=df_PROP_LOAD.mean().apply(numpy.round).values
# PROP_DELAY=df_PROP_DELAY.mean().values
# PROP_delay_std_R = df_PROP_DELAY.std().values
# PROP_USAGE = df_PROP_USAGE.mean().values
# PROP_usage_std_R = df_PROP_USAGE.std().values
# PROP_THOUGHPUT = df_PROP_throughput.mean().values
# PROP_throughput_std_R = df_PROP_throughput.std().values

plt.figure()

#DELAy
title = "{0}_ONU,{1}*SEED,max_grant={2},bucket={3},distance={4},max_buffer={5}".format(NUMBER_OF_ONUs,TIMES_SEED,sys.argv[5],sys.argv[4],sys.argv[3],sys.argv[6])
plt.title(title)
plt.xlabel("load (%)")
plt.ylabel("delay (s)")
plt.fill_between(IPACT_LOAD, IPACT_DELAY - IPACT_delay_std,IPACT_DELAY + IPACT_delay_std, alpha=0.1,color="r")

plt.fill_between(PERF_LOAD, PERF_DELAY - PERF_delay_std,PERF_DELAY + PERF_delay_std, alpha=0.1,color="b")
#plt.fill_between(PROP_LOAD, PROP_DELAY - PROP_delay_std_R,PROP_DELAY + PROP_delay_std_R, alpha=0.1,color="g")
plt.plot(IPACT_LOAD, IPACT_DELAY, 'o-', color="r",label="IPACT")
plt.plot(PERF_LOAD, PERF_DELAY, '>-', color="b",label="PRED_Perfeita")
#plt.plot(PROP_LOAD, PROP_DELAY, '+-', color="g",label="Pred_proposta")
plt.legend(loc='upper center', shadow=True)
plt.savefig("Delay-"+title)

#USAGE
plt.figure()
plt.xlabel("load (%)")
plt.ylabel("utilization (%)")
plt.fill_between(IPACT_LOAD, IPACT_USAGE - IPACT_usage_std,IPACT_USAGE + IPACT_usage_std, alpha=0.1,color="r")
plt.fill_between(PERF_LOAD, PERF_USAGE - PERF_usage_std,PERF_USAGE + PERF_usage_std, alpha=0.1,color="b")
#plt.fill_between(PROP_LOAD, PROP_USAGE - PROP_usage_std_R,PROP_USAGE + PROP_usage_std_R, alpha=0.1,color="g")
plt.plot(IPACT_LOAD, IPACT_USAGE, 'o-', color="r",label="IPACT")
plt.plot(PERF_LOAD, PERF_USAGE, '>-', color="b",label="PRED_Perfeita")
#plt.plot(PROP_LOAD, PROP_USAGE, '+-', color="g",label="Pred_proposta")
plt.legend(loc='upper center', shadow=True)
plt.savefig("utilization-"+title)

#THROUGHPUT
plt.figure()
plt.xlabel("load (%)")
plt.ylabel("throughput (%)")
plt.fill_between(IPACT_LOAD, IPACT_TRHOUGHPUT - IPACT_throughput_std,IPACT_TRHOUGHPUT + IPACT_throughput_std, alpha=0.1,color="r")
plt.fill_between(PERF_LOAD, PERF_THROUGHPUT - PERF_throughput_std,PERF_THROUGHPUT + PERF_throughput_std, alpha=0.1,color="b")
#plt.fill_between(PROP_LOAD, PROP_THOUGHPUT - PROP_throughput_std_R,PROP_THOUGHPUT + PROP_throughput_std_R, alpha=0.1,color="g")
plt.plot(IPACT_LOAD, IPACT_TRHOUGHPUT, 'o-', color="r",label="IPACT")
plt.plot(PERF_LOAD, PERF_THROUGHPUT, '>-', color="b",label="PRED_Perfeita")
#plt.plot(PROP_LOAD, PROP_THOUGHPUT, '+-', color="g",label="Pred_proposta")

plt.legend(loc='upper center', shadow=True)
plt.savefig("Throughput-"+title)
#plt.show()
#print REQUEST[-6:]
# print("average pkt delay: %s" % (numpy.mean(Delay)))
# print("channel idle: %s%%" % (100-(100 * numpy.sum(UTILIZATION)/float(SIM_DURATION))))
# rate = (numpy.sum(PKT)*8)/(float(SIM_DURATION)*float(1000000000))
# print("upstream utilization: %s%%" % (100*(rate)))

import simpy
import random
import functools
import time

SIM_DURATION = 0.05
RANDOM_SEED = 42
REQUEST = []
NUMBER_OF_ONUs = 3


class Cable(object):
    """This class represents the propagation through a cable."""
    def __init__(self, env, delay):
        self.env = env
        #self.delay = delay
        self.upstream = simpy.Store(env)

    def latency(self, value,delay):
        yield self.env.timeout(delay)
        self.upstream.put(value)

    def put(self, value,delay):
        self.env.process(self.latency(value,delay))

    def get(self):
        return self.upstream.get()

#yield store.put('spam %s' % i)
#print('Produced spam at', env.now)
def DBA_IPACT(RTT,buffer):
    sending_time = buffer*8/1000.0
    grant_time = RTT + sending_time + 0.0001
    return grant_time

# def ONU_sender(env, cable, ONU):
#     """A process which randomly generates messages."""
#     while True:
#         #pkt_size = random.randint(5,20)
#         #yield ONU.queue.put('pkt_size %s' % pkt_size)
#         #print('pkt queued at %.6f' % env.now)
#
#
#         # wait for next transmission
#         yield env.timeout(0.0001)
#
#         #port.out = ps
#         q_size =ONU.port.byte_size
#         print("ONU %s queue is %s" % (ONU.oid,q_size))
#         #print(ONU.port.store.items)
#         msg = {'text':"ONU %s sent this REQUEST for %.6f at %.6f" %
#             (ONU.oid,ONU.port.byte_size, env.now),'queue_size':q_size,'ONU':ONU}
#         cable.put((msg),ONU.delay)
#         ONU.port.set_grant(q_size)
#         sent = env.process(ONU.port.sent())
#         yield sent


# def OLT_receiver(env, cable,olt):
#     """A process which consumes messages."""
#     while True:
#         # Get event for message pipe
#         #pkt_size = random.randint(5,20)
#         msg = yield cable.get()
#         print('OLT Received this at %.6f while %s' % (env.now, msg['text']))
#         REQUEST.append((msg['ONU'].oid,msg['queue_size'],env.now))
#         print msg['ONU'].delay
#         dba = olt.DBA_IPACT(msg['ONU'].delay,msg['queue_size'])
#         yield env.timeout(dba)


        #self.queue = simpy.Store(self.env, capacity=20)

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
            p = Packet(self.env.now, self.sdist(), self.packets_sent, src=self.id, flow_id=self.flow_id)
            self.out.put(p)

class PacketSink(object):

    def __init__(self, env, rec_arrivals=False, absolute_arrivals=False, rec_waits=True, debug=False, selector=None):
        self.store = simpy.Store(env)
        self.env = env
        self.rec_waits = rec_waits
        self.rec_arrivals = rec_arrivals
        self.absolute_arrivals = absolute_arrivals
        self.waits = []
        self.arrivals = []
        self.debug = debug
        self.action = env.process(self.run())  # starts the run() method as a SimPy process
        self.packets_rec = 0
        self.bytes_rec = 0
        self.selector = selector

    def run(self):
        last_arrival = 0.0
        while True:
            msg = (yield self.store.get())
            if not self.selector or self.selector(msg):
                now = self.env.now
                if self.rec_waits:
                    self.waits.append(self.env.now - msg.time)
                if self.rec_arrivals:
                    if self.absolute_arrivals:
                        self.arrivals.append(now)
                    else:
                        self.arrivals.append(now - last_arrival)
                    last_arrival = now
                self.packets_rec += 1
                self.bytes_rec += msg.size
                if self.debug:
                    print msg

    def put(self, pkt):
        self.store.put(pkt)

class SwitchPort(object):


    def __init__(self, env, rate, qlimit=None, debug=False):
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
        self.debug = debug
        self.busy = 0  # Used to track if a packet is currently being sent
        self.action = env.process(self.run())  # starts the run() method as a SimPy process

    def set_grant(self,grant_size):
        self.grant_size = grant_size

    def sent(self):
        while self.grant_size > 0:
            msg = (yield self.store.get())
            self.busy = 1
            self.byte_size -= msg.size
            self.grant_size -= msg.size
            print "ENVIANDO"
            yield self.env.timeout(msg.size*8.0/self.rate)


    def run(self):
        while True:
            with self.res.request() as req:
                yield req
                yield self.env.timeout(0.2)
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
            #return self.store.put(pkt)

class ONU(object):
    def __init__(self,distance,oid,env,cable):
        self.env = env
        self.distance = distance
        self.oid = oid
        self.delay = self.distance/ float(210000)
        self.thread_delay = 0
        #self.queue = simpy.Store(self.env, capacity=20)
        adist = functools.partial(random.expovariate, 50)
        sdist = functools.partial(random.expovariate, 0.01)  # mean size 100 bytes
        samp_dist = functools.partial(random.expovariate, 1.0)
        port_rate = 1000.0
        #ps = PacketSink(env, debug=False, rec_arrivals=True)
        self.pg = PacketGenerator(self.env, "Greg", adist, sdist)
        self.port = SwitchPort(self.env, port_rate, qlimit=10000)
        self.pg.out = self.port
        self.sender = self.env.process(self.ONU_sender(cable))

    def ONU_sender(self, cable):
        """A process which randomly generates messages."""
        while True:
            #pkt_size = random.randint(5,20)
            #yield ONU.queue.put('pkt_size %s' % pkt_size)
            #print('pkt queued at %.6f' % env.now)


            # wait for next transmission
            yield self.env.timeout(0.0001)

            #port.out = ps
            q_size =self.port.byte_size
            print("ONU %s queue is %s" % (self.oid,q_size))
            #print(ONU.port.store.items)
            msg = {'text':"ONU %s sent this REQUEST for %.6f at %.6f" %
                (self.oid,self.port.byte_size, self.env.now),'queue_size':q_size,'ONU':self}
            cable.put((msg),self.delay)
            self.port.set_grant(q_size)
            sent = self.env.process(self.port.sent())
            yield sent

class OLT(object):
    def __init__(self,env,cable):
        self.env = env
        self.guard_int = 0.0001
        self.receiver = self.env.process(self.OLT_receiver(cable))

    def DBA_IPACT(self,RTT,buffer):
        sending_time = buffer*8/1000.0
        grant_time = RTT + sending_time + self.guard_int
        return grant_time

    def OLT_receiver(self,cable):
        """A process which consumes messages."""
        while True:
            # Get event for message pipe
            #pkt_size = random.randint(5,20)
            msg = yield cable.get()
            print('OLT Received this at %.6f while %s' % (self.env.now, msg['text']))
            REQUEST.append((msg['ONU'].oid,msg['queue_size'],self.env.now))
            print msg['ONU'].delay
            dba = self.DBA_IPACT(msg['ONU'].delay,msg['queue_size'])
            yield self.env.timeout(dba)


# Setup and start the simulation
print('Event Latency')
random.seed(RANDOM_SEED)
env = simpy.Environment()


cable = Cable(env, 10)
ONU_List = []
for i in range(NUMBER_OF_ONUs):
    ONU_List.append(ONU(100,i,env,cable))
#onu0 = ONU(20,0,env)
#onu1 = ONU(100,1,env,cab)
olt = OLT(env,cable)
#env.process(ONU_sender(env, cable, onu0))
#env.process(ONU_sender(env, cable, onu1))
#env.process(OLT_receiver(env, cable,olt))

env.run(until=SIM_DURATION)
print REQUEST[-6:]

import numpy as np

#Import Process level primitives
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

from lava.proc.lif.process import LIF
from lava.proc.dense.process import LearningDense, Dense
from lava.proc.sparse.process import LearningSparse, Sparse
from lava.proc.conv.process import Conv 


from lava.proc.monitor.process import Monitor

from lava.magma.core.run_configs import Loihi1SimCfg

from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.model.py.model import PyLoihiProcessModel

# Import ProcessModel ports, data-types
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType

# Import execution protocol and hardware resources
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU

# Import decorators
from lava.magma.core.decorator import implements, requires

from lava.magma.core.run_conditions import RunSteps

# Import STDP learning rule
from lava.proc.learning_rules.stdp_learning_rule import STDPLoihi
from lava.magma.core.learning.learning_rule import Loihi2FLearningRule

import tonic
from lava.proc.conv import utils
from scipy.sparse import coo_matrix, coo_array

# Input events for our network.
# coordinates are [x, y, polarity, time]
# Here we could load a pre-existing event dataset
sensor_size = (16, 16)


dataset = tonic.datasets.DVSGesture(save_to='./', train=True)
raw_events, label = dataset[0]

events = np.zeros((len(raw_events), 4))
for e in range(len(raw_events)):
    events[e][0] = int(raw_events[e][0]*(sensor_size[0]/128))
    events[e][1] = int(raw_events[e][1]*(sensor_size[1]/128))
    events[e][2] = int(raw_events[e][2])
    events[e][3] = int(raw_events[e][3]/1000)
events = events.astype(int)

events = events
events[:, 3]+=3
#print(events)

events = np.array(events)
# Classes to feed our events to an SNN
class SpikeInput(AbstractProcess):
    """Takes an array of events and converts it to input spikes"""

    def __init__(self, vth: int, events : np.ndarray):
        super().__init__()
        shape = (sensor_size[0]*sensor_size[1],)
        self.spikes_out = OutPort(shape=shape)  # Input spikes to the SNN
        self.v = Var(shape=shape, init=0)
        self.vth = Var(shape=(1,), init=vth)
        self.events = Var(shape=(len(events), 4), init=events)

@implements(proc=SpikeInput, protocol=LoihiProtocol)
@requires(CPU)
class PySpikeInputModel(PyLoihiProcessModel):
    spikes_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    v: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    vth: int = LavaPyType(int, int, precision=32)
    events: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    
    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)

    def post_guard(self):
        """Guard function for PostManagement phase.
        """
        return True

    def run_post_mgmt(self):
        """Post-Management phase: executed only when guard function above 
            returns True.
        """
        self.v = np.zeros(self.v.shape)

    def run_spk(self):
        """Spiking phase: executed unconditionally at every time-step
        """
        events_at_this_timestep = self.events[self.events[:, 3]==self.time_step]
        for e in events_at_this_timestep:
            self.v[e[0]*sensor_size[0]+e[1]] += self.vth	
        s_out1 = self.v >= self.vth
        s_out2 = np.argwhere(s_out1==True)
        self.spikes_out.send(data=s_out1)

class OutputProcess(AbstractProcess):
    """Process to gather spikes from output LIF neurons"""

    def __init__(self, **kwargs):
        super().__init__()
        shape = (sensor_size[0]*sensor_size[1],)
        self.spikes_in = InPort(shape=shape)
        self.spikes_accum = Var(shape=shape)  # Accumulated spikes for classification

@implements(proc=OutputProcess, protocol=LoihiProtocol)
@requires(CPU)
class PyOutputProcessModel(PyLoihiProcessModel):
    spikes_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    spikes_accum: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=32)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)

    def post_guard(self):
        """Guard function for PostManagement phase.
        """
        return False

    def run_post_mgmt(self):
        """Post-Management phase: executed only when guard function above
        returns True.
        """
        pass

    def run_spk(self):
        """Spiking phase: executed unconditionally at every time-step
        """
        spk_in = self.spikes_in.recv()
        self.spikes_accum = self.spikes_accum + spk_in
# Create processes
spike_input = SpikeInput(vth=1, events=events)
#Here we define our weights for our dense layer to perform lateral inhibition
def calculate_distance(pos1, pos2):
    """
    Calculate the Euclidean distance between two points.
    """
    return np.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)

def initialize_wta_weights(sensor_width, sensor_height, w_max=50):
    weights = np.zeros((sensor_width * sensor_height, sensor_width * sensor_height))
    
    # Iterate over each 'neuron' in the 2D grid
    for i in range(sensor_width * sensor_height):
        for j in range(sensor_width * sensor_height):
            # Convert the linear index to 2D coordinates
            pos_i = (i % sensor_width, i // sensor_width)
            pos_j = (j % sensor_width, j // sensor_width)
            
            # Calculate the Euclidean distance between neurons
            d = calculate_distance(pos_i, pos_j)
            
            # Apply the WTA weight formula
            weights[i, j] = min(np.exp(d) / (sensor_width * sensor_height), w_max)
    
    return weights

# Initialize the weights
weights = initialize_wta_weights(sensor_size[0],sensor_size[1])
# Here we define a sparse matrix corresponding to the convolution operation and use it to build a convolutional layer
sparse_conv = utils.conv_to_sparse(input_shape=(sensor_size[0], sensor_size[1], 1),
                                output_shape=(sensor_size[0], sensor_size[1], 1), 
                                kernel=np.full((1, 1, 1, 1), 5), 
                                stride=(1, 1), 
                                padding=(0, 0), 
                                dilation=(1, 1), 
                                group=1)
#print(sparse_conv)

mat = coo_matrix((sparse_conv[2].astype(int), 
                  (sparse_conv[0].astype(int), 
                  sparse_conv[1].astype(int))), 
                  shape=(256, 256))
#sparse = Sparse(weights=mat)
sparse = LearningSparse(weights=mat,
                        learning_rule=STDP)
dense = Dense(weights=weights,     # WTA weights         
                name='dense')
lif1 = LIF(shape=(sensor_size[0]*sensor_size[1],),                         # Number and topological layout of units in the process
           vth=0.5,                             # Membrane threshold
           dv=0.5,                              # Inverse membrane time-constant
           du=1.0,                              # Inverse synaptic time-constant
           bias_mant=0,           # Bias added to the membrane voltage in every timestep 
           name="lif1")
output_proc = OutputProcess()

# Connect Processes
spike_input.spikes_out.connect(sparse.s_in)
sparse.a_out.connect(dense.s_in)
dense.a_out.connect(lif1.a_in)
lif1.s_out.connect(output_proc.spikes_in)

# Monitor states of our lif neurons
monitor_lif1 = Monitor()
num_steps = np.max(events[:, 3])+1
monitor_lif1.probe(lif1.v, num_steps)

# Run condition : we only need to run our network for a few timesteps
run_condition = RunSteps(num_steps=num_steps)

# Run config : we use CPU
run_cfg = Loihi1SimCfg(select_tag="floating_pt")

# Running simulation
lif1.run(condition=run_condition, run_cfg=run_cfg)

# Visualize stuff
data_lif1 = monitor_lif1.get_data()
import matplotlib
from matplotlib import pyplot as plt

# Create a subplot for each monitor
fig = plt.figure(figsize=(16, 5))
ax0 = fig.add_subplot(121)


# Plot the recorded data
monitor_lif1.plot(ax0, lif1.v) # < Here we visualize voltages, but we could also simply register spikes
plt.show()

print("Accumulated spikes:", np.array(output_proc.spikes_accum))
print(output_proc.spikes_accum.get()[2])
print(np.shape(output_proc.spikes_accum.get()))
plt.imshow(np.reshape(output_proc.spikes_accum.get(), sensor_size))
plt.show()

# Stop the execution
lif1.stop()






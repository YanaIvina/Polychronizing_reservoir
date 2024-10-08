from collections import namedtuple
from tqdm import tqdm
import numpy as np
import pandas as pd
import nest
import random

from .generic_transformer_class import Generic_spiking_transformer
from .utils import (
    generate_random_state,
    convert_neuron_ids_to_indices,
    convert_random_parameters_to_nest
)
from .common_model_components import disable_plasticity


nest.set_verbosity('M_QUIET')

Network_objects_tuple = namedtuple(
    'Network_objects_tuple',
    (
        'exc_reservoir_neuron_ids',
        'inh_reservoir_neuron_ids',
        'input_neuron_ids_A',
        'input_neuron_ids_B',
        'input_neuron_ids_C',
        'input_neuron_ids_D',
        'generators_ids_A',
        'generators_ids_B',
        'generators_ids_C',
        'generators_ids_D',
        'all_connection_descriptors',
        'input_to_reservoir_connections',
        'spike_recorder_id',
        'multimeter_0', 'multimeter_1',
        'voltmeter',
        'wr',
    )
)

def encode_data_to_spike_times(X, network_parameters):
    return np.reshape(
        np.asarray(X) * network_parameters['one_vector_duration'],
        (-1, 1)
    )

class MyTopologyFailed(Exception):
    pass

def create_random_topology(n_neurons, n_connections, neurons):
    while True:
        prepost = []
        try:
            pre_neurons = neurons
            post_neurons = neurons
            conn_counters = [0 for i in range(len(post_neurons))]

            for pre in pre_neurons:
                post_for_current_pre = []
                for min_connections in range(n_connections):
                    while len(post_for_current_pre) < n_connections:
                        s = [
                            post_neurons[i]
                            for i in range(n_neurons)
                            if conn_counters[i] <= min_connections
                            and post_neurons[i] != pre
                            and post_neurons[i] not in post_for_current_pre
                            and not (len(prepost) > i and (pre in prepost[i])) 
                         ]       
                        if len(s) == 0:
                            # No neurons remain with <= min_connections
                            # incoming connections.
                            # Will try increasing min_connections,
                            # thus searching among "less vacant" neurons.
                            break
                        post_chosen = random.choice(s)
                        post_for_current_pre.append(post_chosen)
                        conn_counters[
                            post_neurons.index(post_chosen)
                         ] += 1

                prepost += [post_for_current_pre]

                if len(post_for_current_pre) < n_connections:
                     raise MyTopologyFailed
    
        except MyTopologyFailed:
          # continue the outermost while,
          # i. e. "while true"
          continue
        # If the exception was not raised,
        # break the endless while.
        return prepost
        break

class PaugamMoisyNetwork(Generic_spiking_transformer):

    def __init__(
        self,
        network_parameters,
        neuron_parameters,
        synapse_parameters,
        random_state=None,
        n_jobs=1,
        warm_start=False,
        experiment = None,
        nA = None, nC = None, nB = None, nD = None,
        topology = None,
    ):
        self.network_parameters = network_parameters
        self.neuron_parameters = neuron_parameters
        self.synapse_parameters = synapse_parameters
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.warm_start = warm_start
        self.experiment = experiment
        self.nA = nA
        self.nB = nB
        self.nC = nC
        self.nD = nD
        self.topology = topology

    def _create_network(self, testing_mode):
        number_of_inputs = self.n_features_in_
        n_threads = self.n_jobs
        random_state = (
            self.random_state if not self.random_state is None
            else generate_random_state()
        )

        # Make a copy because we will tamper with neuron_parameters
        # if creating neurons with pre-recorded thresholds.
        # Also, run nest.CreateParameter on those parameters
        # that are dictionaries describing random distributions. 
        neuron_parameters, synapse_parameters = map(
            convert_random_parameters_to_nest,
            (self.neuron_parameters, self.synapse_parameters)
        )

        if testing_mode:
            # Disable synaptic plasticity.
            synapse_parameters = disable_plasticity(
                synapse_parameters
            )

        # Remove existing NEST objects if any exist.
        nest.ResetKernel()
        nest.SetKernelStatus({
            'resolution': 0.1,
            'local_num_threads': n_threads,
        })
        nest.rng_seed = random_state
        random.seed(a = random_state)
        # Create nodes.

        generators_ids_A = nest.Create('spike_generator', number_of_inputs)
        generators_ids_B = nest.Create('spike_generator', number_of_inputs)
        generators_ids_C = nest.Create('spike_generator', number_of_inputs)
        generators_ids_D = nest.Create('spike_generator', number_of_inputs)
        input_neuron_ids_A = nest.Create('parrot_neuron', number_of_inputs)
        input_neuron_ids_B = nest.Create('parrot_neuron', number_of_inputs)
        input_neuron_ids_C = nest.Create('parrot_neuron', number_of_inputs)
        input_neuron_ids_D = nest.Create('parrot_neuron', number_of_inputs)
        exc_reservoir_neuron_ids = nest.Create(
            'iaf_psc_delta',
            self.network_parameters['number_of_exc_neurons'],
            params=neuron_parameters
        )
        inh_reservoir_neuron_ids = []#nest.Create(
        #     'iaf_psc_delta',
        #     self.network_parameters['number_of_inh_neurons'],
        #     params=neuron_parameters
        # )

        spike_recorder_id = nest.Create('spike_recorder')
        nest.Connect(exc_reservoir_neuron_ids, spike_recorder_id, conn_spec='all_to_all')

        multimeter_0 = nest.Create('multimeter', params={'record_from': ['V_m']})
        multimeter_1 = nest.Create('multimeter', params={'record_from': ['V_m']})
        nest.Connect(multimeter_0, exc_reservoir_neuron_ids[0])
        nest.Connect(multimeter_1, exc_reservoir_neuron_ids[1])

        voltmeter = nest.Create("voltmeter")
        nest.Connect(voltmeter, exc_reservoir_neuron_ids[0])

        wr = nest.Create('weight_recorder')
        nest.CopyModel('stdp_tanh_synapse', 'stdp_tanh_synapse_rec', {"weight_recorder": wr})

        # Create connections.
        # ------------------
        # Static synapses from the generators
        # to parrot neurons representing the input.
        nest.Connect(pre=generators_ids_A, post=input_neuron_ids_A, conn_spec='one_to_one', syn_spec='static_synapse')
        nest.Connect(pre=generators_ids_B, post=input_neuron_ids_B, conn_spec='one_to_one', syn_spec='static_synapse')
        nest.Connect(pre=generators_ids_C, post=input_neuron_ids_C, conn_spec='one_to_one', syn_spec='static_synapse')
        nest.Connect(pre=generators_ids_D, post=input_neuron_ids_D, conn_spec='one_to_one', syn_spec='static_synapse')

        exp = self.experiment
        topology = self.topology
        nA = self.nA
        nB = self.nB
        nC = self.nC
        nD = self.nD
        # self.r= exc_reservoir_neuron_ids[list(nB)]

        synapse = {
            'synapse_model': 'static_synapse',
            'weight': 1.0,
            'delay': 0.5,
        }
        internal_indegree_and_outdegree = 3
        neuron_ids = exc_reservoir_neuron_ids

        nest.Connect(
            pre = input_neuron_ids_A,
            post = exc_reservoir_neuron_ids[list(nA)],
            conn_spec = {
                'rule': 'all_to_all',
            },
            syn_spec = synapse
        )
        nest.Connect(
            pre = input_neuron_ids_B,
            post = exc_reservoir_neuron_ids[list(nB)],
            conn_spec = {
                'rule': 'all_to_all',
            },
            syn_spec = synapse
        )
        nest.Connect(
            pre = input_neuron_ids_C,
            post = exc_reservoir_neuron_ids[list(nC)],
            conn_spec = {
                'rule': 'all_to_all',
            },
            syn_spec = synapse
        )
        nest.Connect(
            pre = input_neuron_ids_D,
            post = exc_reservoir_neuron_ids[list(nD)],
            conn_spec = {
                'rule': 'all_to_all',
            },
            syn_spec = synapse
        )

        # create_topology(exp,input_neuron_ids_A,input_neuron_ids_B,input_neuron_ids_C,input_neuron_ids_D,exc_reservoir_neuron_ids,synapse)

        if topology == 'irreg':
            prepost = create_random_topology(n_neurons = 10, n_connections = internal_indegree_and_outdegree, neurons = neuron_ids.global_id)
            self.p = min(neuron_ids.global_id)
            self.pr = prepost
            post = []
            post += [prepost[j][i] for i in range(internal_indegree_and_outdegree) for j in range(len(neuron_ids))]
            post1 = [post[i:i + len(neuron_ids)] for i in range(0, len(post), len(neuron_ids))]
            for index_shift in range(internal_indegree_and_outdegree):
                nest.Connect(
                    pre = neuron_ids,
                    post = post1[index_shift],
                    conn_spec='one_to_one',
                    syn_spec=self.synapse_parameters
                )
        if topology == 'reg':
            for index_shift in range(1, internal_indegree_and_outdegree+1):
                nest.Connect(
                    pre=neuron_ids,
                    post=[
                        neuron_ids[
                            (neuron_number + index_shift) % len(neuron_ids)
                        ].global_id
                        for neuron_number in range(len(neuron_ids))
                    ],
                    conn_spec='one_to_one',
                    syn_spec= self.synapse_parameters
                )

##################################################################

        # nest.Connect(
        #     pre=inh_reservoir_neuron_ids,
        #     post=exc_reservoir_neuron_ids + inh_reservoir_neuron_ids,
        #     conn_spec={
        #         'rule': 'pairwise_bernoulli',
        #         'p': 0.3,
        #         'allow_autapses': False,
        #     },
        #     syn_spec={
        #         'synapse_model': 'stdp_paugam_moisy_synapse',
        #         'weight': -0.5,
        #         'Wmax': -1.,
        #     }
        # )

        # nest.Connect(exc_reservoir_neuron_ids, spike_recorder_id, conn_spec='all_to_all')
        # nest.Connect(
        #     (
        #         input_neuron_ids_1
        #         + exc_reservoir_neuron_ids
        #     ),
        #     spike_recorder_id,
        #     conn_spec='all_to_all'
        # )
##################################################################

        # Now that all connections have been created,
        # request their descriptors from NEST.
        all_connection_descriptors = {
            'all_internal': nest.GetConnections(
                source=exc_reservoir_neuron_ids,
                target=exc_reservoir_neuron_ids
            )}

        nest.SetStatus(all_connection_descriptors['all_internal'], 'delay', 10.0)

        input_neuron = input_neuron_ids_A + input_neuron_ids_B + input_neuron_ids_C + input_neuron_ids_D
        input_to_reservoir_connections = nest.GetConnections(
            source=input_neuron,
            target=exc_reservoir_neuron_ids
        )
        input_to_reservoir_connections = pd.DataFrame(
            convert_neuron_ids_to_indices(
                np.asarray(
                    nest.GetStatus(
                        input_to_reservoir_connections,
                        'weight'
                    )
                ),
                input_to_reservoir_connections,
                input_neuron,
                exc_reservoir_neuron_ids
            )
        )
        
        self.network_objects = Network_objects_tuple(
            exc_reservoir_neuron_ids=exc_reservoir_neuron_ids,
            inh_reservoir_neuron_ids=inh_reservoir_neuron_ids,
            generators_ids_A=generators_ids_A,
            generators_ids_B=generators_ids_B,
            generators_ids_C=generators_ids_C,
            generators_ids_D=generators_ids_D,
            input_neuron_ids_A=input_neuron_ids_A,
            input_neuron_ids_B=input_neuron_ids_B,
            input_neuron_ids_C=input_neuron_ids_C,
            input_neuron_ids_D=input_neuron_ids_D,
            all_connection_descriptors=all_connection_descriptors,
            input_to_reservoir_connections=input_to_reservoir_connections,
            spike_recorder_id=spike_recorder_id,
            multimeter_0=multimeter_0,
            multimeter_1=multimeter_1,
            voltmeter=voltmeter,
            wr=wr,
        )

    def run_the_simulation(self, A, B, C, D, y_train):
        """ Encode X into input spiking rates
        and feed to the network.

        Learning is unsupervised: y_train is only used to indicate
        testing if y_train is None and training otherwise.
        If testing:
        * The simulation duration is fixed at 1 epoch.
        * STDP is disabled.
        * Re-normalization of weights is not applied even if enabled.
        * Output spiking rates are recorded and returned.
        If training:
        * Weights are saved to self.weights_.
        * Nothing is returned.
        """
        testing_mode = y_train is None
        n_epochs = self.network_parameters['epochs'] if not testing_mode else 1

        progress_bar = tqdm(total=n_epochs * len(A))

        # exp = self.experiment
        # if(exp in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8]):
        #     for epoch in range(n_epochs):

        #         for x in X:
        #             spike_times = (
        #                 nest.biological_time
        #                 + encode_data_to_spike_times(x, self.network_parameters)
        #             )

        #             # The simulation itself.
        #             nest.SetStatus(
        #                 self.network_objects.generators_ids,
        #                 'spike_times',
        #                 spike_times
        #             )

        #             nest.Simulate(self.network_parameters['intervector_duration'])
        #             progress_bar.update()   

        # if(exp in [1.9, 1.7,2.0,2.1,2.2,2.3,2.4,2.5,2.6,3.0,3.1,3.2,3.3,3.4,3.5,3.6,4.0,4.1,4.2,4.3,4.4,4.5,4.6,5.0,5.1,5.2,5.3,5.4,5.5,5.6]):  
        
        for epoch in range(n_epochs):

            for x in A:
                spike_times = (
                    nest.biological_time
                    + encode_data_to_spike_times(x, self.network_parameters)
                )
                nest.SetStatus(self.network_objects.generators_ids_A, 'spike_times', spike_times) 

            for x in B:
                spike_times = (
                    nest.biological_time
                    + encode_data_to_spike_times(x, self.network_parameters)
                )
                nest.SetStatus(self.network_objects.generators_ids_B, 'spike_times', spike_times)

            for x in C:
                spike_times = (
                    nest.biological_time
                    + encode_data_to_spike_times(x, self.network_parameters)
                )
                nest.SetStatus(self.network_objects.generators_ids_C, 'spike_times', spike_times)

            for x in D:
                spike_times = (
                    nest.biological_time
                    + encode_data_to_spike_times(x, self.network_parameters)
                )
                nest.SetStatus(self.network_objects.generators_ids_D, 'spike_times', spike_times)

            nest.Simulate(self.network_parameters['intervector_duration'])
            progress_bar.update() 

        progress_bar.close()


        neuron_type_mapping = {}
        neuron_type_mapping.update({
            neuron_id: 'excitatory'
            for neuron_id in self.network_objects.exc_reservoir_neuron_ids.global_id
        })
        # neuron_type_mapping.update({
        #     neuron_id: 'inhibitory'
        #     for neuron_id in self.network_objects.inh_reservoir_neuron_ids.global_id
        # })
        # neuron_type_mapping.update({
        #     neuron_id: 'input'
        #     for neuron_id in self.network_objects.input_neuron_ids.global_id
        # })
        spike_raster = pd.DataFrame(nest.GetStatus(self.network_objects.spike_recorder_id, keys='events')[0])
        spike_raster['neuron_type'] = spike_raster['senders'].map(neuron_type_mapping)
        self.spike_raster_ = spike_raster

        V_m = pd.DataFrame(nest.GetStatus(self.network_objects.multimeter_1,'events')[0])
        self.V_m_ = V_m

        conn_w = pd.DataFrame(nest.GetStatus(self.network_objects.wr,'events')[0])
        self.conn_w_ = conn_w

        weights = np.asarray(
            nest.GetStatus(
                self.network_objects.all_connection_descriptors['all_internal'],
                'weight')
        )
        weights = convert_neuron_ids_to_indices(
            weights,
            self.network_objects.all_connection_descriptors['all_internal'],
            self.network_objects.exc_reservoir_neuron_ids,
            self.network_objects.exc_reservoir_neuron_ids
        )          
        self.weights_ = weights
        
        # Empty the detector.
        nest.SetStatus(self.network_objects.spike_recorder_id, {'n_events': 0})
        nest.SetStatus(self.network_objects.multimeter_0, {'n_events': 0})
        nest.SetStatus(self.network_objects.wr, {'n_events': 0})


        # V_reset = nest.GetStatus(self.network_objects.exc_reservoir_neuron_ids[0], ["V_reset"])[0][0]
        # V_th = nest.GetStatus(self.network_objects.exc_reservoir_neuron_ids[0], ["V_th"])[0][0]       
        # v = self.network_objects.voltmeter.get("events", "V_m")
        # V_0 = nest.GetStatus(self.network_objects.multimeter_0)[0]['events']['V_m']
        # V_1 = nest.GetStatus(self.network_objects.multimeter_1)[0]['events']['V_m']
        # t = nest.GetStatus(self.network_objects.multimeter_0)[0]['events']['times']
        # V_t = pd.DataFrame({'t': np.array(t, dtype=object),
        #                     'V_0': np.array(V_0, dtype=object),
        #                     'V_1': np.array(V_1, dtype=object),
        #                     'V_reset': [V_reset for i in range(len(V_0))],
        #                     'V_th': [V_th for i in range(len(V_0))],})

        # s = nest.GetStatus(self.network_objects.sr)
        # self.st_ = spike_raster
        # conn = nest.GetConnections(
        #         source=self.network_objects.exc_reservoir_neuron_ids,
        #         target=self.network_objects.exc_reservoir_neuron_ids
        #     )
        
        

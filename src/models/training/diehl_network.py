from collections import namedtuple
from tqdm import tqdm
import numpy as np
import nest

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
        'exc_neuron_ids',
        'inh_neuron_ids',
        'generators_ids',
        'inputs_ids',
        'all_connection_descriptors',
        'exc_neurons_spike_recorder_id',
        'inh_neurons_spike_recorder_id',
    )
)

def encode_data_to_spike_rates(X, network_parameters):
    return X * (network_parameters['high_rate'] - network_parameters['low_rate']) + network_parameters['low_rate']


class DiehlNetwork(Generic_spiking_transformer):

    def __init__(
        self,
        network_parameters,
        neuron_parameters,
        synapse_parameters,
        random_state=None,
        n_jobs=1,
        warm_start=False
    ):
        self.network_parameters = network_parameters
        self.neuron_parameters = neuron_parameters
        self.synapse_parameters = synapse_parameters
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.warm_start = warm_start

    def _create_network(self, testing_mode):
        number_of_inputs = self.n_features_in_
        n_threads = self.n_jobs
        random_state = (
            self.random_state if not self.random_state is None
            else generate_random_state()
        )
        create_spike_recorders = testing_mode

        # Make a copy because we will tamper with neuron_parameters
        # if creating neurons with pre-recorded thresholds.
        # Also, run nest.CreateParameter on those parameters
        # that are dictionaries describing random distributions. 
        neuron_parameters, synapse_parameters = map(
            convert_random_parameters_to_nest,
            (self.neuron_parameters, self.synapse_parameters)
        )

        if testing_mode:
            # Disable dynamic threshold.
            neuron_parameters['exc_neurons']['Theta_plus'] = 0.
            # Disable synaptic plasticity.
            synapse_parameters['input_to_exc'] = disable_plasticity(
                synapse_parameters['input_to_exc']
            )
        if hasattr(self, 'exc_neurons_thresholds_'):
            # Set the recorded dynamic threshold values.
            neuron_parameters['exc_neurons'] = [
                # because, unlike synapse parameters, NEST only accepts
                # setting varying neuron parameters
                # by passing a dictionary per each neuron.
                dict(
                    neuron_parameters['exc_neurons'],
                    V_th=V_th
                )
                for V_th in exc_neurons_thresholds
            ]

        # Remove existing NEST objects if any exist.
        nest.ResetKernel()
        nest.SetKernelStatus({
            'resolution': 0.1,
            'local_num_threads': n_threads,
        })
        nest.rng_seed = random_state

        # Create nodes.
        exc_neuron_ids = nest.Create(
            'iaf_cond_exp_adaptive',
            self.network_parameters['number_of_exc_neurons'],
            params=neuron_parameters['exc_neurons']
        )
        inh_neuron_ids = nest.Create(
            'iaf_cond_exp_adaptive',
            self.network_parameters['number_of_inh_neurons'],
            params=neuron_parameters['inh_neurons']
        )
        generators_ids = nest.Create('poisson_generator', number_of_inputs)
        inputs_ids = nest.Create('parrot_neuron', number_of_inputs)
        if create_spike_recorders:
            exc_neurons_spike_recorder_id = nest.Create('spike_recorder')
            inh_neurons_spike_recorder_id = nest.Create('spike_recorder')

        populations_to_connect = [
            ('input_to_exc', inputs_ids, exc_neuron_ids),
            ('input_to_inh', inputs_ids, inh_neuron_ids),
            ('exc_to_inh', exc_neuron_ids, inh_neuron_ids),
            ('inh_to_exc', inh_neuron_ids, exc_neuron_ids)
        ]

        # Create connections.
        # ------------------
        # Static synapses from the generators
        # to parrot neurons representing the input.
        nest.Connect(pre=generators_ids, post=inputs_ids, conn_spec='one_to_one', syn_spec='static_synapse')
        # Connections from the input-representing parrot neurons
        # to excitatory and inhibitory neurons.
        if hasattr(self, 'weights_'):
            # Re-create connections from the saved weights.
            for exc_or_inh, exc_or_inh_neuron_ids in (
                ('exc', exc_neuron_ids),
                ('inh', inh_neuron_ids)
            ):
                conn_type_name = 'input_to_' + exc_or_inh
                sparse_weight_array = self.weights_[conn_type_name]
                
                synapse_parameters[conn_type_name].update(
                    weight=sparse_weight_array['weight'],
                    delay=sparse_weight_array['delay']
                )
                nest.Connect(
                    pre=np.array(inputs_ids)[sparse_weight_array['pre_index']],
                    post=np.array(exc_neuron_ids)[sparse_weight_array['post_index']],
                    conn_spec='one_to_one',
                    syn_spec=synapse_parameters[conn_type_name]
                )
        else:
            # Plastic synapses from the input parrot neurons
            # to the excitatory neurons.
            nest.Connect(
                pre=inputs_ids,
                post=exc_neuron_ids,
                conn_spec='all_to_all',
                syn_spec=synapse_parameters['input_to_exc']
            )
            # Static synapses from the input parrot neurons
            # to the inhibitory neurons.
            nest.Connect(
                pre=inputs_ids,
                post=inh_neuron_ids,
                conn_spec={
                    'rule': 'fixed_total_number',
                    'N': int(
                        self.network_parameters['input_to_inh_connection_prob']
                        * len(inputs_ids)
                        * len(inh_neuron_ids)
                    )
                },
                syn_spec=synapse_parameters['input_to_inh']
            )
        # Static connections from excitatory neurons
        # to their inhibitory counterparts.
        nest.Connect(
            pre=exc_neuron_ids,
            post=inh_neuron_ids,
            conn_spec='one_to_one',
            syn_spec=synapse_parameters['exc_to_inh']
        )
        # Static connections from inhibitory neurons
        # to excitatory ones.
        for current_neuron_number in range(self.network_parameters['number_of_inh_neurons']):
            nest.Connect(
                pre=inh_neuron_ids[current_neuron_number:current_neuron_number+1],
                post=nest.NodeCollection(list(
                    set(exc_neuron_ids.tolist())
                    - set(exc_neuron_ids[current_neuron_number:current_neuron_number+1].tolist())
                )),
                conn_spec='all_to_all',
                syn_spec=synapse_parameters['inh_to_exc']
            )
        # Connect neurons to spike detectors.
        if create_spike_recorders:
            nest.Connect(exc_neuron_ids, exc_neurons_spike_recorder_id, conn_spec='all_to_all')
            nest.Connect(inh_neuron_ids, inh_neurons_spike_recorder_id, conn_spec='all_to_all')

        # Now that all connections have been created,
        # request their descriptors from NEST.
        all_connection_descriptors = {
            conn_type_name: nest.GetConnections(source=pre_ids, target=post_ids)
            for conn_type_name, pre_ids, post_ids in populations_to_connect
        }

        self.network_objects = Network_objects_tuple(
            exc_neuron_ids=exc_neuron_ids,
            inh_neuron_ids=inh_neuron_ids,
            generators_ids=generators_ids,
            inputs_ids=inputs_ids,
            all_connection_descriptors=all_connection_descriptors,
            exc_neurons_spike_recorder_id=exc_neurons_spike_recorder_id if create_spike_recorders else None,
            inh_neurons_spike_recorder_id=inh_neurons_spike_recorder_id if create_spike_recorders else None,
        )


    def run_the_simulation(self, X, y_train):
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
        input_spike_rates = encode_data_to_spike_rates(X, self.network_parameters)
        record_weights = not testing_mode
        record_spikes = testing_mode

        progress_bar = tqdm(
            total=n_epochs * len(input_spike_rates)
        )
        for epoch in range(n_epochs):
            if record_spikes:
                output_spiking_rates = {
                    'exc_neurons': [],
                    'inh_neurons': []
                }
            for x in input_spike_rates:
                # Weight normalization
                if (not testing_mode
                    and not self.network_parameters['weight_normalization_during_training'] is None
                ):
                    for neuron_id in self.network_objects.exc_neuron_ids:
                        this_neuron_input_synapses = nest.GetConnections(
                            source=self.network_objects.inputs_ids,
                            target=[neuron_id]
                        )
                        w = nest.GetStatus(this_neuron_input_synapses, 'weight')
                        w = np.array(w) * self.network_parameters['weight_normalization_during_training'] / sum(w)
                        nest.SetStatus(this_neuron_input_synapses, 'weight', w)

                # The simulation itself.
                nest.SetStatus(self.network_objects.generators_ids, [{'rate': r} for r in x])
                nest.Simulate(self.network_parameters['one_vector_longtitude'])

                nest.SetStatus(self.network_objects.generators_ids, {'rate': 0.})
                nest.Simulate(self.network_parameters['intervector_pause'])

                if record_spikes:
                    for neuron_type_name, neurons_ids, spike_recorder_id in (
                        ('exc_neurons', self.network_objects.exc_neuron_ids, self.network_objects.exc_neurons_spike_recorder_id),
                        ('inh_neurons', self.network_objects.inh_neuron_ids, self.network_objects.inh_neurons_spike_recorder_id),
                    ):
                        # NEST returns all_spikes == {
                        #   'times': spike_times_array,
                        #   'senders': senders_ids_array
                        # }
                        all_spikes = nest.GetStatus(spike_recorder_id, keys='events')[0]
                        current_input_vector_output_rates = [
                            # 1000.0 * len(all_spikes['times'][
                            #   all_spikes['senders'] == current_neuron
                            # ]) / self.network_parameters['one_vector_longtitude']
                            len(all_spikes['times'][
                                all_spikes['senders'] == current_neuron
                            ])
                            for current_neuron in neurons_ids
                        ]
                        output_spiking_rates[neuron_type_name].append(
                            current_input_vector_output_rates
                        )
                        # Empty the detector.
                        nest.SetStatus(spike_recorder_id, {'n_events': 0})
                progress_bar.update()
        progress_bar.close()
        if record_weights:
            exc_neuron_ids = self.network_objects.exc_neuron_ids
            inh_neuron_ids = self.network_objects.inh_neuron_ids
            generators_ids = self.network_objects.generators_ids
            inputs_ids = self.network_objects.inputs_ids
            all_connection_descriptors = self.network_objects.all_connection_descriptors

            weights_of_all_connection_types = {
                conn_type_name: convert_neuron_ids_to_indices(
                    weights=nest.GetStatus(all_connection_descriptors[conn_type_name], 'weight'),
                    delays=nest.GetStatus(all_connection_descriptors[conn_type_name], 'delay'),
                    connection_descriptors=all_connection_descriptors[conn_type_name],
                    pre_neuron_ids=pre_ids,
                    post_neuron_ids=post_ids
                )
                for conn_type_name, pre_ids, post_ids in (
                    ('input_to_exc', inputs_ids, exc_neuron_ids),
                    ('input_to_inh', inputs_ids, inh_neuron_ids),
                    ('exc_to_inh', exc_neuron_ids, inh_neuron_ids),
                    ('inh_to_exc', inh_neuron_ids, exc_neuron_ids)
                )
            }
            self.weights_ = weights_of_all_connection_types

        if record_spikes:
            return output_spiking_rates['exc_neurons']

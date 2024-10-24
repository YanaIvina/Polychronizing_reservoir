from collections import namedtuple
from tqdm import tqdm
import numpy as np
import nest

from .generic_transformer_class import Generic_spiking_transformer
from .common_model_components import disable_plasticity
from .utils import (
    generate_random_state,
    convert_neuron_ids_to_indices,
    convert_random_parameters_to_nest
)


nest.set_verbosity('M_QUIET')

Network_objects_tuple = namedtuple(
    'Network_objects_tuple',
    (
        'neuron_ids',
        'generators_ids',
        'inputs_ids',
        'all_connection_descriptors',
        'spike_recorder_id'
    )
)

def encode_data_to_spike_rates(X, network_parameters):
    return X * (network_parameters['high_rate'] - network_parameters['low_rate']) + network_parameters['low_rate']

class ClasswiseNetwork(Generic_spiking_transformer):
    def __init__(
        self,
        network_parameters,
        neuron_parameters,
        synapse_parameters,
        random_state=None,
        early_stopping=True,
        n_jobs=1,
        warm_start=False
    ):
        self.network_parameters = network_parameters
        self.neuron_parameters = neuron_parameters
        self.synapse_parameters = synapse_parameters
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.n_jobs = n_jobs
        self.warm_start = warm_start

    def _create_network(self, testing_mode):
        number_of_inputs = self.n_features_in_
        random_state = (
            self.random_state if not self.random_state is None
            else generate_random_state()
        )
        number_of_classes = len(self.classes_)
        create_spike_recorders = testing_mode
        # Make a copy because we will tamper with synapse_parameters
        # if creating connections with pre-recorded weights.
        # Also, run nest.CreateParameter on those parameters
        # that are dictionaries describing random distributions. 
        neuron_parameters, synapse_parameters = map(
            convert_random_parameters_to_nest,
            (self.neuron_parameters, self.synapse_parameters)
        )
        if testing_mode:
            synapse_parameters = disable_plasticity(synapse_parameters)

        # Remove existing NEST objects if any exist.
        nest.ResetKernel()
        n_threads = self.n_jobs
        nest.SetKernelStatus({
            'resolution': 1,
            'local_num_threads': n_threads,
        })
        nest.rng_seed = random_state

        neuron_ids = nest.Create(
            self.network_parameters['neuron_model'],
            number_of_classes,
            params=self.neuron_parameters
        )
        generators_ids = nest.Create('poisson_generator', number_of_inputs)
        inputs_ids = nest.Create('parrot_neuron', number_of_inputs)
        if create_spike_recorders:
            spike_recorder_id = nest.Create('spike_recorder')

        # Create connections.
        nest.Connect(
            pre=generators_ids,
            post=inputs_ids,
            conn_spec='one_to_one',
            syn_spec='static_synapse'
        )
        if (
            hasattr(self, 'weights_')
            and (
                testing_mode
                # Continue fitting with existing weights.
                or self.warm_start
            )
        ):
            synapse_parameters.update(weight=self.weights_['weight'])
            nest.Connect(
                pre=np.array(inputs_ids)[self.weights_['pre_index']],
                post=np.array(neuron_ids)[self.weights_['post_index']],
                conn_spec='one_to_one',
                syn_spec=synapse_parameters
            )        
        else:
            nest.Connect(
                pre=inputs_ids,
                post=neuron_ids,
                conn_spec='all_to_all',
                syn_spec=synapse_parameters
            )
        if create_spike_recorders:
            nest.Connect(neuron_ids, spike_recorder_id, conn_spec='all_to_all')

        # Now that all connections have been created,
        # request their descriptors from NEST.
        all_connection_descriptors = nest.GetConnections(source=inputs_ids, target=neuron_ids)
        self.network_objects = Network_objects_tuple(
            neuron_ids=neuron_ids,
            generators_ids=generators_ids,
            inputs_ids=inputs_ids,
            all_connection_descriptors=all_connection_descriptors,
            spike_recorder_id=spike_recorder_id if create_spike_recorders else None
        )


    def run_the_simulation(self, X, y_train=None):
        testing_mode = y_train is None
        n_epochs = self.network_parameters['epochs'] if not testing_mode else 1
        input_spike_rates = encode_data_to_spike_rates(X, self.network_parameters)
        record_weights = not testing_mode
        record_spikes = testing_mode
        early_stopping = self.early_stopping and not testing_mode

        progress_bar = tqdm(
            total=n_epochs * len(input_spike_rates)
        )
        if early_stopping:
            previous_weights = np.asarray(
                [-1] * len(self.network_objects.all_connection_descriptors)
            )
        for epoch in range(n_epochs):
            if record_spikes:
                output_spiking_rates = []
            for vector_number, x, in enumerate(input_spike_rates):
                # The simulation itself.
                nest.SetStatus(self.network_objects.generators_ids, [{'rate': r} for r in x])
                if not testing_mode:
                    y = y_train[vector_number]
                    nest.SetStatus(
                        self.network_objects.neuron_ids,
                        [
                            {
                                # Inject negative stimulation current
                                # into all neurons that do not belong
                                # to the current class, so that to
                                # prevent them from spiking
                                # (and thus from learning
                                # the current class).
                                'I_e': 0. if current_neuron == y else -1e+3,
                                # That current may have made the neuron's
                                # potential too negative.
                                # We reset the potential, so that previous
                                # stimulation not inhibit spiking
                                # in response to the current input.
                                'V_m': self.neuron_parameters['E_L'],
                            }
                            for current_neuron in self.classes_
                        ]
                    )
                nest.Simulate(self.network_parameters['one_vector_longtitude'])

                if record_spikes:
                    # NEST returns all_spikes == {
                    #   'times': spike_times_array,
                    #   'senders': senders_ids_array
                    # }
                    all_spikes = nest.GetStatus(self.network_objects.spike_recorder_id, keys='events')[0]
                    current_input_vector_output_rates = [
                        # 1000.0 * len(all_spikes['times'][
                        #   all_spikes['senders'] == current_neuron
                        # ]) / network_parameters['one_vector_longtitude']
                        len(all_spikes['times'][
                            all_spikes['senders'] == current_neuron
                        ])
                        for current_neuron in self.network_objects.neuron_ids
                    ]
                    # Empty the detector.
                    nest.SetStatus(self.network_objects.spike_recorder_id, {'n_events': 0})
                    output_spiking_rates.append(
                        current_input_vector_output_rates
                    )
                progress_bar.update()
            if record_weights or early_stopping:
                weights = np.asarray(
                    nest.GetStatus(self.network_objects.all_connection_descriptors, 'weight')
                )
            if early_stopping:
                if (
                    np.abs(
                        weights - previous_weights
                    ) < 0.001
                ).all():
                    print(
                        'Early stopping because none of the weights'
                        'have changed by more than 0.001 for an epoch.',
                        'This usually means that the neuron emits no spikes.'
                    )
                    break
                if np.logical_or(
                    weights < 0.1,
                    weights > 0.9
                ).all():
                    print('Early stopping on weights convergence to 0 or 1.')
                    break
                previous_weights = weights
        progress_bar.close()

        if record_weights:
            weights = convert_neuron_ids_to_indices(
                weights,
                self.network_objects.all_connection_descriptors,
                self.network_objects.inputs_ids,
                self.network_objects.neuron_ids
            )
            self.weights_ = weights

        if record_spikes:
            return output_spiking_rates

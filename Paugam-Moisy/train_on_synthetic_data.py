import array
from sys import path as pythonpath
from subprocess import check_output
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import nest

from sys import argv

this_file_path = str(pathlib.Path(__file__).resolve().parent)
repo_root_path = str(
    pathlib.Path(
        check_output(
            '''
                cd {this_file_path}
                echo -n `git rev-parse --show-toplevel`
            '''.format(this_file_path=this_file_path),
            shell=True
        ).decode('utf-8')
    ).resolve()
)

pythonpath.append(repo_root_path)

tau_m = argv[1]
t_ref = argv[2]
delay = float(argv[3])
stdp = argv[4]
U = float(argv[5])
signal = argv[6]
topology = argv[7]
ex = int(argv[8])
conn = int(argv[9])

from src.models.training.paugam_moisy_network import PaugamMoisyNetwork

from signal_parameters import parameters

from model_parameters import (
	V_reset,
	U,
	one_vector_duration,
	intervector_duration,
	n_epochs)

labelsAB, vector, nA,nB,nC,nD, n, x = parameters(signal, ex)

net = PaugamMoisyNetwork(
	n_jobs = 10,
	random_state = 307,
	warm_start = True,
	experiment = ex,
	topology = topology,
	nA = nA, nB = nB, nC = nC, nD = nD,
	network_parameters={
		'one_vector_duration': one_vector_duration,
		'intervector_duration': intervector_duration,
		'number_of_exc_neurons': 10,
		'number_of_inh_neurons': 1,
		'internal_indegree_and_outdegree': conn,
		'epochs': 1,
	},
	neuron_parameters={
		'tau_m': float(tau_m),
		't_ref': float(t_ref),
		'tau_minus': float(stdp),
		'V_th': (-50. - V_reset) / U,
		'E_L': (-65. - V_reset) / U,
		'V_reset': (-65. - V_reset) / U,
		'V_m': (-65. - V_reset) / U,
		'I_e': 0.,
	},
	synapse_parameters={
        'synapse_model': 'stdp_tanh_synapse_rec',
        'weight': [0.5] * 10,
        'Wmax': 1.,
		'a_plus': 0.14,
		'a_minus': -0.12,
        # 'mu_plus': 0.,
		# 'mu_minus': 0.,
		# 'tau_plus': float(stdp),
		# 'alpha': 1.5,
		# 'lambda': 0.01,
		'delay': [delay] * 10,
		# 'tau_plus': float(stdp),
	},
)

net.n_features_in_ = len(x[0])
net._create_network(testing_mode=False)

for epoch in range(n_epochs):
	net.run_the_simulation(A=[x[0]]*2, B=[x[1]]*2, C=[x[2]]*2, D=[x[3]]*2, y_train=[0,1])

	pd.DataFrame(
		net.weights_
	).to_csv(
		f'{signal}-{topology}-{ex}-tau={tau_m}-t_ref={t_ref}-delay={delay}-stdp={stdp}/weights-vector,{epoch}.csv',
		sep='\t', index=False)

	pd.DataFrame(
		net.weights_
	).to_csv(
		f'all-weights-{tau}/{signal}.csv',
		sep='\t', index=False)

	pd.DataFrame(
		net.network_objects.input_to_reservoir_connections
	).to_csv(
		f'{signal}-{topology}-{ex}-tau={tau_m}-t_ref={t_ref}-delay={delay}-stdp={stdp}/input_to_reservoir_connections-vector,{epoch}.csv',
		sep='\t', index=False)
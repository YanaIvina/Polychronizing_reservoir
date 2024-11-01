from sys import path as pythonpath
from subprocess import check_output
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

pythonpath.append('/s/ls4/users/yanaivina/second/spiking_classifiers')

from src.models.training.paugam_moisy_network import PaugamMoisyNetwork
from signal_parameters import parameters
from model_parameters import (V_reset, U, one_vector_duration, intervector_duration, n_epochs)


def Network(search_space):

	tau_m = search_space['tau_m']
	t_ref = search_space['t_ref']
	delay = search_space['delay']
	a_plus = search_space['a_plus']
	a_minus = search_space['a_minus']
	tau_plus = search_space['tau_plus']
	tau_minus = search_space['tau_minus']

	U = 16.0
	signal = 'ABAB'
	topology = 'reg_'
	ex = 3
	conn = 9

	if topology == 'reg_':
		weight = 0.5
		delay1 = delay
		all_conn = conn*10-12
	else:
		weight = [0.5]*10
		delay1 = [delay]*10
		all_conn = conn*10
		
	# for signal in ['AAAA','AAAB','AABA','AABB','ABAA','ABAB','ABBA','ABBB','BAAA','BAAB','BABA','BABB','BBAA','BBAB','BBBA','BBBB']:
	comm = 0
	for signal in ['AAAA','AAAB','AABA','AABB','ABAA','ABAB','ABBA','ABBB','BAAA','BAAB','BABA','BABB','BBAA','BBAB','BBBA','BBBB']:
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
				'tau_m': tau_m,
				't_ref': t_ref,
				'tau_minus': tau_minus,
				'V_th': (-50. - V_reset) / U,
				'E_L': (-65. - V_reset) / U,
				'V_reset': (-65. - V_reset) / U,
				'V_m': (-65. - V_reset) / U,
				'I_e': 0.,
			},
			synapse_parameters={
				'synapse_model': 'stdp_tanh_synapse_rec',
				'weight': weight,
				'Wmax': 1.,
				'a_plus': a_plus,
				'a_minus': a_minus,
				'delay': delay1,
				'tau_plus': tau_plus,
			},
		)

		net.n_features_in_ = len(x[0])
		net._create_network(testing_mode=False)

		# k = 0
		# print(all_conn)
		# w = [0 for i in range(all_conn)]
		for epoch in range(n_epochs):
			net.run_the_simulation(A=[x[0]]*2, B=[x[1]]*2, C=[x[2]]*2, D=[x[3]]*2, y_train=[0,1])
		# 	if all(w == net.weights_['weight']): k+=1
		# 	else: k = 0
		# 	if k==30: break
		
		wx = len(net.weights_['weight'][net.weights_['weight']>0.5])
		wx = abs(wx - (all_conn/2))/(all_conn/2)
		comm += wx
	return comm

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

search_space = {
                'tau_m' : hp.uniform('tau_m', 1, 7),
                't_ref' : hp.uniform('t_ref', 1, 5),
				'delay': hp.uniform('delay', 1, 7),
				'a_plus': hp.uniform('a_plus', 0, 2),
				'a_minus': hp.uniform('a_minus', -2, 0),
				'tau_plus': hp.uniform('tau_plus', 1, 30),
				'tau_minus': hp.uniform('tau_minus', 1, 30),
                }
trials = Trials()

best = fmin( 
            fn=Network,  
            space=search_space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials,
            show_progressbar=True
        )
print(best)
print(trials)

file = open('optimized_parameters.txt', 'wt') 
file.write(str(best)) 
file.close() 
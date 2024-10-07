from sys import path as pythonpath
import pathlib
from subprocess import check_output
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt 

from sys import argv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

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

from model_parameters import n_epochs, one_vector_duration, intervector_duration, n_epochs
from signal_parameters import parameters

tau_m = argv[1]
t_ref = argv[2]
delay = argv[3]
stdp = argv[4]
U = argv[5]
signal = argv[6]
topology = argv[7]
ex = int(argv[8])
Th = float(argv[9])

labelsAB, vector, nA,nB,nC,nD, neurons, x = parameters(signal, ex)

connections = []
for epoch in range(n_epochs):
    connections += [pd.read_csv(
        f'{signal}-{topology}-{ex}-tau={tau_m}-t_ref={t_ref}-delay={delay}-stdp={stdp}/weights-vector,{epoch}.csv',
        sep='\t'
    )]

# РАСЧЁТ КОЛИЧЕСТВА ОДНОМЕРНЫХ, ДВУМЕРНЫХ И ТРЁХМЕРНЫХ СИМПЛЕКСОВ (только для случая с трёмя входящими и трёмя исходящими связями, иначе менять алгоритм)
n_simplex1D, n_simplex2D, n_simplex3D = [0]*n_epochs, [0]*n_epochs, [0]*n_epochs
for epoch in range(n_epochs):
    edge, triangle, pyramid = [], [], []
    for a in range(10):
        syn = []
        for i in range(len(connections[epoch])):
            if (connections[epoch]['pre_index'][i] == a) and (connections[epoch]['weight'][i] >= Th):
                syn += [connections[epoch]['post_index'][i]]
                n_simplex1D[epoch] += 1
                e = [a] + [connections[epoch]['post_index'][i]] 
                edge += [e]
        for i in range(len(connections[epoch])):
            if (connections[epoch]['pre_index'][i] in syn) and (connections[epoch]['post_index'][i] in syn) and (connections[epoch]['weight'][i] >= Th):
                n_simplex2D[epoch] += 1
                t = [a] + [connections[epoch]['pre_index'][i]] + [connections[epoch]['post_index'][i]] 
                triangle += [t]
        if len(syn) == 3:
            for s in syn:
                d = []
                for i in range(len(connections[epoch])):
                    if (connections[epoch]['pre_index'][i] == s) and (connections[epoch]['post_index'][i] in syn) and (connections[epoch]['weight'][i] >= Th):
                        d += [connections[epoch]['post_index'][i]]
                for i in range(len(connections[epoch])):
                    if (len(d) == 2) and (connections[epoch]['pre_index'][i] in d) and (connections[epoch]['post_index'][i] in d) and (connections[epoch]['weight'][i] >= Th):
                        n_simplex3D[epoch] += 1
                        p = [a] + [syn[0]] + [syn[1]] + [syn[2]]
                        pyramid += [p]
    # исключить симплексы, которые входят в состав других симплексов
    if(n_simplex3D[epoch]>0):
        tinp = []
        for n in range(n_simplex2D[epoch]):
            for m in range(n_simplex3D[epoch]):
                if set(triangle[n]).issubset(pyramid[m]):
                    tinp += [triangle[n]]
        triangle = [triangle[i] for i in range(len(triangle)) if triangle[i] not in tinp]
        n_simplex2D[epoch] = len(triangle) 
    if(n_simplex2D[epoch]>0):
        eint = []
        for n in range(n_simplex1D[epoch]):
            for m in range(n_simplex2D[epoch]):
                if set(edge[n]).issubset(triangle[m]):
                    eint += [edge[n]]
        edge = [edge[i] for i in range(len(edge)) if edge[i] not in eint]
        n_simplex1D[epoch] = len(edge)

# записать количество симплексов в файл
# f = open(f'results.txt', 'a')
# f.write(f'{signal} , {vector} , {topology} , {neurons} , {tau_m} , {n_simplex1D[n_epochs-1]} , {n_simplex2D[n_epochs-1]} , {n_simplex3D[n_epochs-1]}\n')
# f.close()

epoch = [e for e in range(n_epochs)]

fig = plt.figure(figsize = (30, 17))

# написать параметры эксперимента рядом с графиком
# plt.figtext(0.85 , 0.45, f'tau_m = {tau_m}\nt_ref = {t_ref}\ndelay = {delay}\nstdp = {stdp}\nduration = {intervector_duration}', fontsize = 32)
gs = GridSpec(ncols = 9, nrows = 1, figure = fig)
ax = fig.add_subplot(gs[0, 0:8], label="1")

# ax = fig.add_subplot()
ax.set_title(signal, size = 40, fontweight = 'bold')
ax.plot(epoch, n_simplex2D, linewidth = 5, color = 'r', label='2D simplexes')
ax.plot(epoch, n_simplex3D, linewidth = 5, color = 'b', label='3D simplexes')

ax.set_xlabel('epoch', size = 32)
ax.set_ylabel('number of simplexes', size = 32)
ax.tick_params(axis='both', size = 20, labelsize = 24)
ax.legend(bbox_to_anchor = (1.01 , 1.01), loc = 'upper left', prop = {'size': 28})
ax.grid()

# сохранить в папке для данного эксперимента
plt.savefig(
    f'{signal}-{topology}-{ex}-tau={tau_m}-t_ref={t_ref}-delay={delay}-stdp={stdp}/images/simplexes_Th{Th}.png',
    dpi=800
)
# сохранить все изображения в одной папке
# plt.savefig(
#     f'ALL_SIMPLEXES-{topology}-{ex}-tau={tau_m}-t_ref={t_ref}-delay={delay}-stdp={stdp}/{signal}_Th{Th}.png',
#     dpi=800
#     )

print(f'number_of_simplexes experiment {ex} tau_m={tau_m} t_ref={t_ref} delay={delay} stdp={stdp} done')
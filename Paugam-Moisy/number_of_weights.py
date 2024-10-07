from sys import path as pythonpath
import pathlib
from subprocess import check_output
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

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

from model_parameters import n_epochs, one_vector_duration, intervector_duration
from signal_parameters import parameters

tau_m = argv[1]
t_ref = argv[2]
delay = argv[3]
stdp = argv[4]
U = argv[5]
signal = argv[6]
topology = argv[7]
ex = int(argv[8])

labelsAB, vector, nA,nB,nC,nD, n, x = parameters(signal, ex)

weights = []
for epoch in range(n_epochs):
    weights += [pd.read_csv(
        f'{signal}-{topology}-{ex}-tau={tau_m}-t_ref={t_ref}-delay={delay}-stdp={stdp}/weights-vector,{epoch}.csv',
        sep='\t'
    )]

wght0, wght1, wght2, wght3, wght4, wght5 = [0]*n_epochs, [0]*n_epochs, [0]*n_epochs, [0]*n_epochs, [0]*n_epochs, [0]*n_epochs

# задать диапазоны весов, которые нужно отразить на графике
for epoch in range(n_epochs):
    for w in weights[epoch]['weight']:
        if w < 0.5:
            wght0[epoch] += 1
        # if w >= 0.1 and w == 0.5:
        #     wght1[epoch] += 1
        # if w >= 0.3 and w < 0.55:
        #     wght2[epoch] += 1
        # if w >= 0.5 and w < 0.7:
        #     wght3[epoch] += 1
        # if w >= 0.7 and w < 0.9:
        #     wght4[epoch] += 1
        if w > 0.5:
            wght5[epoch] += 1

from matplotlib import pyplot as plt 

epoch = [e for e in range(n_epochs)]

fig = plt.figure(figsize = (30, 17))

# написать параметры эксперимента рядом с графиком
# plt.figtext(0.85 , 0.45, f'tau_m = {tau_m}\nt_ref = {t_ref}\ndelay = {delay}\nstdp = {stdp}\nduration = {intervector_duration}', fontsize = 32)
# gs = GridSpec(ncols = 9, nrows = 1, figure = fig)
# ax = fig.add_subplot(gs[0, 0:8], label="1")
ax = fig.add_subplot()

ax.set_title(signal, size = 40, fontweight = 'bold')

#визуализация в соответствии с заданными диапазонами
ax.plot(epoch, wght0, '-.', linewidth = 5, color = 'k', label='веса < 0.5')
# ax.plot(epoch, wght1, '--', linewidth = 5, color = 'k',label ='веса = 0.5')
# ax.plot(epoch, wght2, '--', linewidth = 3, color = 'orange', label ='0.3 <= веса < 0.55')
# ax.plot(epoch, wght3, '--', linewidth = 3, color = 'violet', label ='0.5 <= веса < 0.7')
# ax.plot(epoch, wght4, linewidth = 3, color = 'turquoise', label ='0.7 <= веса < 0.9')
ax.plot(epoch, wght5, linewidth = 5, color = 'k', label='веса > 0.5')

ax.set(ylim=(0, 90))
ax.set_xlabel('Epoch', size = 38)
ax.set_ylabel('Number of edges', size = 38)
ax.tick_params(axis='both', size = 20, labelsize = 24)
# ax.legend(bbox_to_anchor = (1.01 , 1.01), loc = 'upper left', prop = {'size': 38})
ax.legend(loc = 'upper right', prop = {'size': 38})
ax.grid()

# сохранить в папке для данного эксперимента
plt.savefig(
    f'{signal}-{topology}-{ex}-tau={tau_m}-t_ref={t_ref}-delay={delay}-stdp={stdp}/images/number_weights.png',
    dpi=800
    )
# сохранить все изображения в одной папке
# plt.savefig(
#     f'ALL_WEIGHTS-{topology}-{ex}-tau={tau_m}-t_ref={t_ref}-delay={delay}-stdp={stdp}/{signal}.png',
#     dpi=800
#     )

print(f'number_of_weights experiment {ex} tau_m={tau_m} t_ref={t_ref} delay={delay} stdp={stdp} done')
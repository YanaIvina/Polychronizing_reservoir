from sys import path as pythonpath
import pathlib
from subprocess import check_output

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

from model_parameters import n_epochs, intervector_duration
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

connections = []
for epoch in range(n_epochs):
    connections += [pd.read_csv(
        f'{signal}-{topology}-{ex}-tau={tau_m}-t_ref={t_ref}-delay={delay}-stdp={stdp}/weights-vector,{epoch}.csv',
        sep='\t'
    )]

connection = []
for epoch in range(n_epochs):
    connection += [connections[epoch]['weight']]
connection = np.array(connection)
w = np.rot90(connection)

fig = plt.figure(figsize = (13, 9))
plt.figtext(0.85 , 0.5, f'tau_m = {tau_m}\nt_ref = {t_ref}\ndelay = {delay}\nstdp = {stdp}\nduration = {intervector_duration}', fontsize = 18)

plt.pcolormesh(w, cmap = 'Greys', snap = 'True')
plt.title(signal, size = 20, fontweight = 'bold')
plt.xlabel('epoch', size = 18)
plt.colorbar()

# сохранить в папке для данного эксперимента
plt.savefig(
    f'{signal}-{topology}-{ex}-tau={tau_m}-t_ref={t_ref}-delay={delay}-stdp={stdp}/images/heatmap.png',
    dpi=800
)
# сохранить все изображения в одной папке
# plt.savefig(
#     f'ALL_HEATMAPS-{topology}-{ex}-tau={tau_m}-t_ref={t_ref}-delay={delay}-stdp={stdp}/{signal}.png',
#     dpi=800
#     )

print(f'plot_weight_heatmap experiment {ex} tau_m={tau_m} t_ref={t_ref} delay={delay} stdp={stdp} done')
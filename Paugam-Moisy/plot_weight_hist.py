from sys import path as pythonpath
import pathlib
from subprocess import check_output
from matplotlib.gridspec import GridSpec

from sys import argv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from model_parameters import intervector_duration
from signal_parameters import parameters

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
delay = argv[3]
stdp = argv[4]
U = argv[5]
signal = argv[6]
topology = argv[7]
ex = int(argv[8])
epoch = int(argv[9])

labelsAB, vector, nA,nB,nC,nD, n, x = parameters(signal, ex)

connections = pd.read_csv(
    f'{signal}-{topology}-{ex}-tau={tau_m}-t_ref={t_ref}-delay={delay}-stdp={stdp}/weights-vector,{epoch}.csv',
    sep='\t'
)

import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure()

# написать параметры эксперимента рядом с графиком
# plt.figtext(0.83 , 0.5, f'tau_m = {tau_m}\nt_ref = {t_ref}\ndelay = {delay}\nstdp = {stdp}\nduration = {intervector_duration}', fontsize = 14)
# gs = GridSpec(ncols = 9, nrows = 1, figure = fig)
# ax = fig.add_subplot(gs[0, 0:8], label="1")
# ax.hist(connections['weight'])
# ax.set_title(f'{signal}, эпоха {epoch}', fontweight = 'bold')
# ax.set_xlabel('Weight', fontsize = 14)
# ax.set_ylabel('Number of edges', fontsize = 14)

plt.hist(connections['weight'])
plt.title(f'{signal}, эпоха {epoch}', fontweight = 'bold')
plt.xlabel('Weight', fontsize = 14)
plt.ylabel('Number of edges', fontsize = 14)


# сохранить в папке для данного эксперимента
plt.savefig(
    f'{signal}-{topology}-{ex}-tau={tau_m}-t_ref={t_ref}-delay={delay}-stdp={stdp}/images/hist_{epoch}.png',
    dpi=800
)
# сохранить все изображения в одной папке
# plt.savefig(
#     f'ALL_HISTS-{topology}-{ex}-tau={tau_m}-t_ref={t_ref}-delay={delay}-stdp={stdp}/{signal}_{epoch}.png',
#     dpi=800
#     )

print(f'plot_weight_hist experiment {ex} tau_m={tau_m} t_ref={t_ref} delay={delay} stdp={stdp} epoch {epoch} done')
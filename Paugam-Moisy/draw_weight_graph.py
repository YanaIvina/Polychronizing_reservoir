from sys import path as pythonpath
import pathlib
from subprocess import check_output

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
Th = float(argv[9])
epoch = int(argv[10])

labelsAB, vector, nA,nB,nC,nD, n, x = parameters(signal, ex)

connections = pd.read_csv(
    f"{signal}-{topology}-{ex}-tau={tau_m}-t_ref={t_ref}-delay={delay}-stdp={stdp}/weights-vector,{epoch}.csv", 
    sep='\t')

G = nx.DiGraph()
G.add_nodes_from(connections['pre_index'].sort_values())

connections = connections[connections['weight'] > Th]
pd.options.mode.chained_assignment = None

fig = plt.figure()
plt.figtext(0.005, 0.5, f'tau_m = {tau_m}\nt_ref = {t_ref}\ndelay = {delay}\nstdp = {stdp}\nduration = {intervector_duration}', fontsize = 7)
plt.title(f'{signal} эпоха {epoch}', fontweight = 'bold')

G.add_edges_from(
    (row.pre_index, row.post_index, {'weight': row.weight})
    for row in connections.itertuples()
)

nx.draw_networkx(G, pos = nx.circular_layout(G), arrows=True)

pos = nx.circular_layout(G)
shift = [0.08, 0.08]
shifted_pos ={node: node_pos + shift for node, node_pos in pos.items()}

nx.draw_networkx_labels(G, shifted_pos, labels=labelsAB, font_color='red', horizontalalignment="left")

plt.figtext(0.8, 0.15, signal, color='red')
plt.title(signal, fontweight = 'bold')

# сохранить в папке для данного эксперимента
plt.savefig(
    f'{signal}-{topology}-{ex}-tau={tau_m}-t_ref={t_ref}-delay={delay}-stdp={stdp}/images/graph_Th{Th},epoch{epoch}.png',
    dpi=800
)
# # сохранить все изображения в одной папке
# plt.savefig(
#     f'ALL_GRAPHS-{topology}-{ex}-tau={tau_m}-t_ref={t_ref}-delay={delay}-stdp={stdp}/{signal}-Th={Th}-epoch={epoch}.png',
#     dpi=800
# )

print(f'draw_weight_graph experiment {ex} tau_m={tau_m} t_ref={t_ref} delay={delay} stdp={stdp} Th={Th} epoch {epoch} done')
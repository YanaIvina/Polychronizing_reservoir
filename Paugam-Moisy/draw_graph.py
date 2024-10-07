from sys import path as pythonpath
import pathlib
from subprocess import check_output

from sys import argv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

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

labelsAB, vector, nA,nB,nC,nD, n, x = parameters(signal, ex)

connections = pd.read_csv(
    f"{signal}-{topology}-{ex}-tau={tau_m}-t_ref={t_ref}-delay={delay}-stdp={stdp}/weights-vector,0.csv", 
    sep='\t')
input_to_reservoir_connections = pd.read_csv(
    f"{signal}-{topology}-{ex}-tau={tau_m}-t_ref={t_ref}-delay={delay}-stdp={stdp}/input_to_reservoir_connections-vector,0.csv", 
    sep='\t')

pd.options.mode.chained_assignment = None

fig = plt.figure()
G = nx.DiGraph()

G.add_nodes_from(connections['pre_index'].sort_values())
G.add_edges_from(
    (row.pre_index, row.post_index, {'weight': row.weight})
    for row in connections.itertuples())

nx.draw_networkx(G, pos = nx.circular_layout(G), arrows=True)
pos = nx.circular_layout(G)
shift = [0.08, 0.08]
shifted_pos = {node: node_pos + shift for node, node_pos in pos.items()}
nx.draw_networkx_labels(G, shifted_pos, labels=labelsAB, font_color='red', horizontalalignment="left")

plt.figtext(0.8, 0.15, signal, color='red')
plt.title(vector, fontweight = 'bold')

plt.savefig(
    f'{signal}-{topology}-{ex}-tau={tau_m}-t_ref={t_ref}-delay={delay}-stdp={stdp}/images/Graph.png',
    dpi=800
)

print(f'draw_graph experiment {ex} tau_m={tau_m} t_ref={t_ref} delay={delay} stdp={stdp} done')

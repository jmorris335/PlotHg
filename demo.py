from constrainthg.hypergraph import Hypergraph, Node
import constrainthg.relations as R
import random

from plothg import plot_simulation, PlotSettings

random.seed(2)
num_nodes = 200
num_edges = num_nodes * 3

alph = range(ord('A'), ord('Z')+1)
labels = [chr(i) + chr(j) + chr(k) for i in alph for j in alph for k in alph]
labels = labels[:num_nodes]

nodes = [Node(l) for l in labels[:num_nodes]]
rels = [R.Rsum, R.Rmean, R.Rmultiply, R.Rmax, R.Rmin]

hg = Hypergraph()
for node in nodes:
    hg.add_node(node)
for i in range(num_edges):
    num_sources = max(1, random.binomialvariate(5, 0.4))
    edge_nodes = random.sample(nodes, num_sources+1)
    source_nodes, target = edge_nodes[:-1], edge_nodes[-1]
    rel = random.choice(rels)
    hg.add_edge(source_nodes, target, rel)

inputs = {random.choice(labels) : random.randrange(20) for i in range(28)}

ps = PlotSettings()
plot_simulation(hg, ps, inputs, random.choice(nodes), search_depth=3000)

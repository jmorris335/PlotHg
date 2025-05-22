from constrainthg.hypergraph import Hypergraph, Edge, Node, TNode
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from functools import partial
import random
import logging

random.seed(2)

class PlotSettings:
    """Settings for visualization.
    
    To configure, instantiate the defaults using ``__init``, then update
    the specific dictionary.

    Example::python

        ps = PlotSettings()
        ps.node_solved.update(facecolor = #ffff00)
    
    """
    def __init__(self):
        self.node_default = dict(
            facecolor = '#ababab',
            edgecolor = 'k',
            linewidth = 0,
            linestyle = '-',
            radius = 0.2,
            zorder = 30,
        )

        self.node_solved = dict(
            facecolor = '#00ffff',
            zorder = 32,
        )

        self.node_source = dict(
            facecolor = '#00aaff',
            zorder = 33,
        )

        self.node_target = dict(
            facecolor = '#bb5555',
            zorder = 35,
        )

        self.node_input = dict(
            facecolor = '#00ffff',
            edgecolor = 'k',
            linewidth = 2.0,
            zorder = 40,
        )

        self.node_output = dict(
            facecolor = '#00bb66',
            edgecolor = 'k',
            linewidth = 0.0,
            zorder = 50,
        )

        self.node_output_solved = self.node_output | dict(
            linewidth = 2.0,
        )

        self.edge_default = dict(
            color = '#cccccc44',
            linewidth = 1,
            zorder = 10,
        )

        self.edge_solved = dict(
            color = '#00118888',
            linewidth = 2,
            zorder = 15,
        )

        self.edge_active = dict(
            color = '#0033cc',
            linewidth = 2,
            zorder = 16,
        )

        self.spacing = dict(
            x_spacing = 0.8,
            y_spacing = 0.8,
            num_rows = 15,
            jitter_rate = 0.1
        )

        self.settings = dict(
            figsize = (10, 8),
            layout = 'constrained'
        )

def plot_simulation(hg: Hypergraph, ps: PlotSettings, inputs: dict, output: Node, **kwargs):
    """Animates a simulation of the hypergraph.
    
    Parameters
    ----------
    hg : Hypergraph
        The hypergraph to simulate.
    ps : PlotSettings
        Settings for the plot.
    inputs : dict
        Inputs to pass to ``hg.solve``, of the form {label : value}.
    output : Node | str
        The node (or node label) to solve for.
    **kwargs
        Other arguments to pass to the solver. See documentation for 
        `Hypergraph.solve() <https://constrainthg.readthedocs.io/en/latest/constrainthg.html#constrainthg.hypergraph.Hypergraph.solve>`_ 
        for more information.
    """
    fig, ax = plt.subplots(**ps.settings)
    tnodes = sim_hg(hg, inputs, output, **kwargs)
    circles, lines = initialize_hg(hg, ax, inputs, output, ps)
    ani = animate_hg(fig, ax, tnodes, list(inputs.keys()), str(output), 
                     circles, lines, ps)
    show_plot(ax)

def show_plot(ax: Axes):
    """Configures and displays the plot."""
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.show()  

def sim_hg(hg: Hypergraph, inputs: dict, target: Node, **kwargs)-> list:
    """Simulates the hypergraph and returns every found TNode."""
    hg.memory_mode = True
    end = hg.solve(target, inputs, **kwargs)
    if end is None:
        raise Exception("No solutions found")
    
    tnodes = trim_unneeded_tnodes(hg, target)
    
    return tnodes

def trim_unneeded_tnodes(hg: Hypergraph, target: Node)-> list:
    """Removes all TNodes after the one representing the successful search."""
    tnodes = hg.solved_tnodes
    target_label = hg.get_node(target).label

    target_index = next((i for i, node in enumerate(tnodes) 
                         if node.node_label == target_label), -1)
    
    tnodes = tnodes[:target_index+1]
    return tnodes

def initialize_hg(hg: Hypergraph, ax: Axes, inputs: dict, output: Node, ps: PlotSettings):
    """Initializes the Hypergraph."""
    circles = plot_nodes(hg, ax, inputs, output, ps)
    lines = plot_edges(hg, ax, circles, ps)
    return circles, lines

def plot_nodes(hg: Hypergraph, ax: Axes, inputs: dict, output: Node, ps: PlotSettings)-> dict:
    """Adds the nodes to the axes as patches."""
    index = 0
    plotted_nodes = {}

    for input in inputs:
        center, index = get_next_center(ps, index)
        plotted_nodes[input] = ax.add_patch(plot_circle('node_input', ps, center))
    
    unplotted = [label for label in hg.nodes if label not in inputs]
    unplotted.remove(str(output))
    curr_node = list(inputs.keys())[-1]

    while len(unplotted) > 0:
        center, index = get_next_center(ps, index)
        curr_node = pop_close_node(curr_node, hg, unplotted)
        circle = plot_circle('node_default', ps, center)
        plotted_nodes[curr_node] = ax.add_patch(circle)

    x_centers, y_centers = zip(*[circle.center for circle in plotted_nodes.values()])
    output_center = (max(x_centers) + ps.spacing['x_spacing'], 
                     sum(y_centers) / len(y_centers))
    output_circle = plot_circle('node_output', ps, output_center)
    # output_circle = plot_output(plotted_nodes, str(output), ps)
    plotted_nodes[str(output)] = ax.add_patch(output_circle)

    return plotted_nodes

def get_next_center(ps: PlotSettings, index: int=0):
    """Returns the next center for the object."""
    x_space, y_space = ps.spacing['x_spacing'], ps.spacing['y_spacing'], 
    height = ps.spacing['num_rows']

    jiggle = (x_space + y_space) / 2 * 0.1 
    y = (index % height) * y_space + (jiggle * random.choice((1, -1)))
    x = (index // height) * x_space + (jiggle * random.choice((1, -1)))

    index += 1

    return (x, y), index
    
def get_random_center(spread, gap, old_center: list=None)-> tuple:
    """Gets a random new spot close to the original (x,y) coordinates."""
    if old_center is None:
        return [0., 0.]
    center = []
    for c in old_center:
        displacement = random.choice((1, -1)) * (random.random() * spread + gap)
        center.append(c + displacement)
    center[0] = abs(center[0])
    return center

def pop_close_node(node_label: str, hg: Hypergraph, unplotted: list)->Node:
    """Returns a close-ish node to the node with the given label and 
    removes the node from unplotted."""
    node = hg.get_node(node_label)
    target_node = get_a_target_of_node(node, hg, unplotted)
    if target_node is not None:
        out_node = target_node
    else:
        source_node = get_a_source_of_node(node, hg, unplotted)
        if source_node is not None:
            out_node = source_node
        else:
            out_node = hg.get_node(random.choice(unplotted))
    out = out_node.label
    unplotted.remove(out)
    return out

def get_a_target_of_node(node: Node, hg: Hypergraph, unplotted: list)->Node:
    """Returns the first encountered node for which the node with the 
    given node_label is a source node."""
    for edge in node.leading_edges:
        target = edge.target
        if target in unplotted:
            return target
    return None

def get_a_source_of_node(node: Node, hg: Hypergraph, unplotted: list)->Node:
    """Returns the first encountered node for which the node with the 
    given node_label is a target node."""
    for edge in node.generating_edges:
        for sn in edge.source_nodes.values():
            if getattr(sn, 'label', '') in unplotted:
                return sn
    return None

def plot_edges(hg: Hypergraph, ax: Axes, plotted_nodes: dict, 
               _plot_settings: dict)->list:
    """Plots the edges of the hypergraph.
    
    :returns: dict of plotted lines {label : Line2D}
    """
    lines = {}
    for edge in hg.edges.values():
        line = plot_edge(edge, ax, plotted_nodes, _plot_settings)
        lines[edge.label] = line
    return lines

def plot_edge(edge: Edge, ax: Axes, plotted_nodes: dict, ps: PlotSettings):
    """Adds the edge to the axis."""
    sn_circles = [plotted_nodes[sn.label] for sn in edge.source_nodes.values()]
    target_circle = plotted_nodes[edge.target.label]
    circles = sn_circles + [target_circle]

    # edge_center = [sum([circle.center[i] for circle in circles]) / len(circles) 
    #                for i in range(2)]

    # x_data, y_data = [edge_center[0]], [edge_center[1]]
    # for circle in circles:
    #     x_data.extend([circle.center[0], edge_center[0]])
    #     y_data.extend([circle.center[1], edge_center[1]])

    t_center = target_circle.center
    x_data, y_data = [t_center[0]], [t_center[1]]
    for circle in circles:
        x_data.extend([circle.center[0], t_center[0]])
        y_data.extend([circle.center[1], t_center[1]])
    
    lines = ax.plot(x_data, y_data, **ps.edge_default)
    return lines[0]


###############################################################################
###   ANIMATION   #############################################################
###############################################################################


def animate_hg(fig, ax: Axes, tnodes: list, inputs: list, output: str, 
               circles: dict, lines: dict, ps: PlotSettings):
    """Animates a simulation of the hypergraph."""
    interval = ps.spacing.get('interval', 500 if len(tnodes) < 50 else 100)

    tnodes_iter = iter(tnodes)
    modified_patches = set()

    ani = animation.FuncAnimation(
        fig, partial(color_active_tnode,
                     t_iter=tnodes_iter,
                     inputs=inputs,
                     output=output,
                     circles=circles, 
                     lines=lines,
                     mod_patches = modified_patches,
                     ps=ps),
        frames=len(tnodes)-1, interval=interval, blit=True)
    return ani

def color_active_tnode(frame: int, t_iter, inputs: list, output: str, 
                       circles: dict, lines: dict, 
                       mod_patches: set, ps: PlotSettings)-> list:
    """Colors the path to the TNode in the plot."""
    try:
        t = next(t_iter)
    except StopIteration:
        color_patch('node_output_solved', ps, circles[output])
        return mod_patches
    
    for line in lines.values():
        color_patch('edge_default', ps, line)
    for label in [key for key in circles]:
        circle = circles[label]
        if circle not in mod_patches:
            continue
        if label in inputs:
            color_patch('node_input', ps, circle)
        color_patch('node_solved', ps, circle)

    
    mod_patches.add(color_patch('node_target', ps, circles[t.node_label]))

    if t.gen_edge_label is not None:
        mod_patches.add(color_patch('edge_active', ps, lines[get_line_label(t)]))

        for child in t.children:
            mod_patches.add(color_patch('node_source', ps, circles[child.node_label]))
            mod_patches.update(color_node_children(child, circles, lines, ps))

    return set(mod_patches)

def get_line_label(tnode: TNode)-> str:
    """Returns the label of the edge as stored in the edge."""
    label = tnode.gen_edge_label.split('#')[0]
    return label

def color_node_children(tnode: TNode, circles, lines, ps, seen: list=None)-> list:
    """Recursive caller coloring the children of `tnode` as found nodes."""
    if seen is None: 
        seen = []
    elif tnode.node_label in seen:
        return []
    seen.append(tnode.node_label)
    out = []

    if tnode.gen_edge_label is not None:
        out.append(color_patch('edge_solved', ps, lines[get_line_label(tnode)]))

        for child in tnode.children:
            out.append(color_patch('node_solved', ps, circles[child.node_label]))
            out.extend(color_node_children(child, circles, lines, ps))
    
    return out

def plot_circle(settings_label: str, ps: PlotSettings, center)-> plt.Circle:
    """Creates and returns the Circle according to the values in PlotSettings."""
    props = getattr(ps, settings_label)
    props = ps.node_default | props
    circle = plt.Circle(center, **props)
    return circle

def color_patch(settings_label: str, ps: PlotSettings, patch)-> Patch:
    """Colors and returns the patch according to the values in PlotSettings."""
    props = getattr(ps, settings_label)
    patch.set(**props)
    return patch
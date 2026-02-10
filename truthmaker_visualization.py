"""
truthmaker_visualization.py

Plotly-based visualization for truthmaker semantics posets.

This module converts NetworkX DiGraphs to interactive Plotly visualizations
with features like hover information, click handlers, and various layout options.
"""

import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_plotly_graph(poset,
                       layout='hierarchical',
                       color_by='generation',
                       show_edges=True,
                       node_size=10,
                       title=None):
    """
    Create an interactive Plotly visualization of a truthmaker poset.

    Args:
        poset: TruthmakerPoset object
        layout: Layout algorithm - 'hierarchical', 'force', or 'circular'
        color_by: How to color nodes - 'generation', 'definite', 'diamond', or 'uniform'
        show_edges: Whether to display edges
        node_size: Base size for nodes (will be scaled by degree)
        title: Optional title for the graph

    Returns:
        plotly.graph_objects.Figure
    """
    # Get the NetworkX graph and proposition list
    G = poset.graph
    props = poset.prop_list

    # Compute layout positions
    pos = compute_layout(G, layout)

    # Prepare node data
    node_trace = create_node_trace(G, props, pos, color_by, node_size)

    # Prepare edge data
    if show_edges:
        edge_trace = create_edge_trace(G, pos)
    else:
        edge_trace = go.Scatter(x=[], y=[], mode='lines')

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])

    # Update layout
    fig.update_layout(
        title=dict(text=title or f"Truthmaker Poset ({len(props)} propositions)", font=dict(size=16)),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        width=1000,
        height=800
    )

    return fig


def compute_layout(G, layout_type):
    """
    Compute node positions based on layout algorithm.

    Args:
        G: NetworkX DiGraph
        layout_type: 'hierarchical', 'force', or 'circular'

    Returns:
        dict mapping node_id -> (x, y) position
    """
    if layout_type == 'hierarchical':
        return hierarchical_layout(G)
    elif layout_type == 'force':
        return nx.spring_layout(G, k=0.5, iterations=50)
    elif layout_type == 'circular':
        return nx.circular_layout(G)
    else:
        raise ValueError(f"Unknown layout type: {layout_type}")


def hierarchical_layout(G):
    """
    Create a hierarchical layout based on node levels.

    Uses longest path from source nodes to properly layer a DAG/poset,
    ensuring each node appears below all its predecessors.

    Args:
        G: NetworkX DiGraph

    Returns:
        dict mapping node_id -> (x, y) position
    """
    # Find source nodes (no predecessors)
    sources = [n for n in G.nodes() if G.in_degree(n) == 0]

    if not sources:
        # If no sources, fall back to spring layout
        return nx.spring_layout(G, k=0.5, iterations=50)

    # Compute longest path from sources using topological sort
    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXError:
        # Not a DAG, fall back to spring layout
        return nx.spring_layout(G, k=0.5, iterations=50)

    # Compute level as longest path from any source
    levels = {}
    for node in topo_order:
        if node in sources:
            levels[node] = 0
        else:
            # Level is max of (predecessor level + 1)
            pred_levels = [levels[pred] for pred in G.predecessors(node)]
            levels[node] = max(pred_levels) + 1 if pred_levels else 0

    # Group nodes by level
    level_groups = {}
    for node, level in levels.items():
        if level not in level_groups:
            level_groups[level] = []
        level_groups[level].append(node)

    # Assign positions
    pos = {}
    max_level = max(levels.values()) if levels else 0

    def node_sort_key(node):
        """
        Create a sort key for nodes that may contain None values.

        Handles M-class tuples like (State, State), (None, State), (State, None), (None, None)
        and L-class tuples like (frozenset, frozenset).
        """
        if isinstance(node, tuple):
            # Convert each element to a sortable string
            def elem_to_str(elem):
                if elem is None:
                    return ""  # Sort None before any state
                elif isinstance(elem, frozenset):
                    # L-class antichain: sort by string representation
                    return ",".join(sorted(s.to_string() for s in elem))
                elif hasattr(elem, 'to_string'):
                    return elem.to_string()
                else:
                    return str(elem)
            return tuple(elem_to_str(e) for e in node)
        return str(node)

    for level, nodes in level_groups.items():
        y = level / max(max_level, 1)  # Normalize to [0, 1]
        n_nodes = len(nodes)

        for i, node in enumerate(sorted(nodes, key=node_sort_key)):
            if n_nodes > 1:
                x = i / (n_nodes - 1)
            else:
                x = 0.5
            pos[node] = (x, y)

    return pos


def create_edge_trace(G, pos):
    """
    Create Plotly trace for edges.

    Args:
        G: NetworkX DiGraph
        pos: dict mapping node_id -> (x, y) position

    Returns:
        plotly.graph_objects.Scatter trace
    """
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    return edge_trace


def create_node_trace(G, props, pos, color_by, base_size):
    """
    Create Plotly trace for nodes with hover information.

    Args:
        G: NetworkX DiGraph
        props: list of BilateralPropositions
        pos: dict mapping node_id -> (x, y) position
        color_by: Coloring scheme
        base_size: Base node size

    Returns:
        plotly.graph_objects.Scatter trace
    """
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Get proposition
        prop = props[node]

        # Create hover text
        hover_text = create_hover_text(node, prop, G)
        node_text.append(hover_text)

        # Determine color
        color = get_node_color(node, prop, color_by)
        node_color.append(color)

        # Determine size based on degree
        degree = G.in_degree(node) + G.out_degree(node)
        size = base_size + degree * 2
        node_size.append(size)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale='Viridis',
            line=dict(width=1, color='white')
        )
    )

    return node_trace


def create_hover_text(node_id, prop, G):
    """
    Create rich hover text for a node.

    Args:
        node_id: Node identifier
        prop: BilateralProposition
        G: NetworkX DiGraph

    Returns:
        str: HTML-formatted hover text
    """
    lines = [
        f"<b>Node {node_id}</b>",
        f"{prop.to_string()}",
        "",
        f"<b>Verifiers:</b>",
    ]

    # Handle null verifiers
    if prop.verifiers.is_null:
        lines.append("  (null)")
    else:
        v_antichain_str = ",".join(sorted(s.to_string() for s in prop.verifiers.antichain))
        lines.append(f"  Antichain: {{{v_antichain_str}}}")
        lines.append(f"  Top: {prop.verifiers.top_state.to_string()}")

    lines.extend([
        "",
        f"<b>Falsifiers:</b>",
    ])

    # Handle null falsifiers
    if prop.falsifiers.is_null:
        lines.append("  (null)")
    else:
        f_antichain_str = ",".join(sorted(s.to_string() for s in prop.falsifiers.antichain))
        lines.append(f"  Antichain: {{{f_antichain_str}}}")
        lines.append(f"  Top: {prop.falsifiers.top_state.to_string()}")

    lines.extend([
        "",
        f"<b>Properties:</b>"
    ])

    if prop.is_definite():
        lines.append("  ✓ Definite")
    if prop.is_diamond():
        lines.append("  ✓ Diamond")

    lines.append("")
    lines.append(f"In-degree: {G.in_degree(node_id)}")
    lines.append(f"Out-degree: {G.out_degree(node_id)}")

    return "<br>".join(lines)


def get_node_color(node_id, prop, color_by):
    """
    Determine node color based on coloring scheme.

    Args:
        node_id: Node identifier
        prop: BilateralProposition
        color_by: Coloring scheme

    Returns:
        Color value (number for continuous, string for discrete)
    """
    if color_by == 'generation':
        # Color by node ID (proxy for generation order)
        return node_id
    elif color_by == 'definite':
        return 1 if prop.is_definite() else 0
    elif color_by == 'diamond':
        return 1 if prop.is_diamond() else 0
    elif color_by == 'uniform':
        return 0
    else:
        return node_id


# Add helper method to RegularSet for visualization
def antichain_to_string_helper(antichain):
    """Helper to convert antichain to string for RegularSet"""
    from truthmaker_core import stateset_to_string
    return stateset_to_string(antichain)


# Monkey-patch RegularSet to add antichain_to_string method for visualization
def patch_regularset():
    """Add antichain_to_string method to RegularSet if not present"""
    from truthmaker_core import RegularSet, stateset_to_string

    def antichain_to_string_method(self):
        """Convert antichain to string, handling null case"""
        if self.is_null:
            return "(null)"
        return stateset_to_string(self.antichain)

    if not hasattr(RegularSet, 'antichain_to_string'):
        RegularSet.antichain_to_string = antichain_to_string_method


# Call patch when module is imported
patch_regularset()


def create_comparison_figure(posets, titles=None, layout='hierarchical'):
    """
    Create a side-by-side comparison of multiple posets.

    Args:
        posets: list of TruthmakerPoset objects
        titles: list of titles for each poset
        layout: Layout algorithm to use

    Returns:
        plotly.graph_objects.Figure with subplots
    """
    n_posets = len(posets)
    if titles is None:
        titles = [f"Poset {i+1}" for i in range(n_posets)]

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=n_posets,
        subplot_titles=titles,
        horizontal_spacing=0.05
    )

    for i, poset in enumerate(posets):
        col = i + 1

        # Compute layout
        G = poset.graph
        props = poset.prop_list
        pos = compute_layout(G, layout)

        # Create traces
        edge_trace = create_edge_trace(G, pos)
        node_trace = create_node_trace(G, props, pos, 'generation', 10)

        # Add to subplot
        fig.add_trace(edge_trace, row=1, col=col)
        fig.add_trace(node_trace, row=1, col=col)

    # Update layout
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        height=600,
        width=400 * n_posets
    )

    # Hide axes for all subplots
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

    return fig


def export_to_dot(poset, filename=None):
    """
    Export poset to GraphViz DOT format.

    Args:
        poset: TruthmakerPoset object
        filename: Optional filename to write to (if None, returns string)

    Returns:
        str: DOT format string (if filename is None)
    """
    G = poset.graph
    props = poset.prop_list

    lines = ['digraph G {', '  rankdir = BT;']

    # Add edges with proposition labels
    for edge in G.edges():
        src_prop = props[edge[0]].to_string()
        dst_prop = props[edge[1]].to_string()
        lines.append(f'  "{src_prop}" -> "{dst_prop}";')

    lines.append('}')

    dot_string = '\n'.join(lines)

    if filename:
        with open(filename, 'w') as f:
            f.write(dot_string)
        return None
    else:
        return dot_string


def export_to_csv(poset, filename):
    """
    Export proposition list to CSV format.

    Args:
        poset: TruthmakerPoset object
        filename: CSV filename to write to
    """
    import csv

    props = poset.prop_list

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'Proposition', 'Verifiers', 'Falsifiers', 'Definite', 'Diamond'])

        for i, prop in enumerate(props):
            writer.writerow([
                i,
                prop.to_string(),
                prop.verifiers.to_string(),
                prop.falsifiers.to_string(),
                prop.is_definite(),
                prop.is_diamond()
            ])


def create_dual_hasse_diagram(prop_set, title=None, width=1400, height=800):
    """
    Create a dual Hasse diagram visualization with L-lattice and M-semilattice.

    Args:
        prop_set: PropositionSet or list of BilateralPropositions
        title: Optional title for the visualization
        width: Figure width in pixels
        height: Figure height in pixels

    Returns:
        plotly.graph_objects.Figure
    """
    from truthmaker_core import (
        build_L_lattice, build_M_semilattice,
        group_by_L_class, group_by_M_class,
        get_L_class, get_M_class,
        L_class_to_string, M_class_to_string,
        PropositionSet
    )

    props = prop_set.to_list() if isinstance(prop_set, PropositionSet) else list(prop_set)

    # Build lattice and semilattice structures
    L_lattice = build_L_lattice(props)
    M_semilattice = build_M_semilattice(props)

    # Apply transitive reduction for clean Hasse diagrams
    L_lattice = nx.transitive_reduction(L_lattice)
    M_semilattice = nx.transitive_reduction(M_semilattice)

    # Get equivalence class groupings
    L_groups = group_by_L_class(props)
    M_groups = group_by_M_class(props)

    # Compute layouts for each side
    L_pos = compute_layout(L_lattice, 'hierarchical')
    M_pos = compute_layout(M_semilattice, 'hierarchical')

    # Shift and scale positions
    # Left panel: x from 0 to 0.35
    # Right panel: x from 0.65 to 1.0
    L_pos_shifted = {node: (x * 0.35, y) for node, (x, y) in L_pos.items()}
    M_pos_shifted = {node: (0.65 + x * 0.35, y) for node, (x, y) in M_pos.items()}

    # Create traces
    traces = []

    # 1. L-lattice edges (black)
    L_edge_trace = create_edge_trace_for_classes(L_lattice, L_pos_shifted, color='#333', width=1)
    traces.append(L_edge_trace)

    # 2. M-semilattice edges (black)
    M_edge_trace = create_edge_trace_for_classes(M_semilattice, M_pos_shifted, color='#333', width=1)
    traces.append(M_edge_trace)

    # 3. Red lines connecting L-classes to M-classes (propositions)
    prop_lines_trace = create_proposition_lines(props, L_pos_shifted, M_pos_shifted)
    traces.append(prop_lines_trace)

    # 4. L-lattice nodes
    L_node_trace = create_class_node_trace(
        L_lattice, L_pos_shifted, L_groups,
        class_type='L', color='lightblue', to_string_fn=L_class_to_string
    )
    traces.append(L_node_trace)

    # 5. M-semilattice nodes
    M_node_trace = create_class_node_trace(
        M_semilattice, M_pos_shifted, M_groups,
        class_type='M', color='lightgreen', to_string_fn=M_class_to_string
    )
    traces.append(M_node_trace)

    # Create figure
    fig = go.Figure(data=traces)

    # Update layout
    fig.update_layout(
        title=dict(
            text=title or f"Dual Hasse Diagram ({len(props)} propositions, {len(L_groups)} L-classes, {len(M_groups)} M-classes)",
            font=dict(size=16)
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=40, l=40, r=40, t=60),
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-0.05, 1.05]
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-0.05, 1.05]
        ),
        plot_bgcolor='white',
        width=width,
        height=height,
        # Add annotations for panel labels
        annotations=[
            dict(
                x=0.175, y=1.05, xref='x', yref='paper',
                text='<b>L-Equivalence Classes (Lattice)</b>',
                showarrow=False, font=dict(size=14)
            ),
            dict(
                x=0.825, y=1.05, xref='x', yref='paper',
                text='<b>M-Equivalence Classes (Semilattice)</b>',
                showarrow=False, font=dict(size=14)
            )
        ]
    )

    return fig


def create_edge_trace_for_classes(graph, pos, color='#888', width=0.5):
    """Create edge trace for equivalence class graph"""
    edge_x = []
    edge_y = []

    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    return go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=width, color=color),
        hoverinfo='none',
        mode='lines'
    )


def create_proposition_lines(props, L_pos, M_pos):
    """Create red lines connecting L-classes to M-classes"""
    from truthmaker_core import get_L_class, get_M_class

    line_x = []
    line_y = []
    hover_texts = []

    for prop in props:
        L_class = get_L_class(prop)
        M_class = get_M_class(prop)

        if L_class in L_pos and M_class in M_pos:
            x0, y0 = L_pos[L_class]
            x1, y1 = M_pos[M_class]
            line_x.extend([x0, x1, None])
            line_y.extend([y0, y1, None])
            hover_texts.extend([prop.to_string(), prop.to_string(), ''])

    return go.Scatter(
        x=line_x, y=line_y,
        line=dict(width=1, color='rgba(255, 0, 0, 0.4)'),  # Red with 40% opacity
        hoverinfo='text',
        text=hover_texts,
        mode='lines',
        name='Propositions'
    )


def create_class_node_trace(graph, pos, groups, class_type='L', color='lightblue', to_string_fn=None):
    """Create node trace for equivalence class graph"""
    node_x = []
    node_y = []
    node_text = []
    node_size = []

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Create hover text
        class_str = to_string_fn(node) if to_string_fn else str(node)
        props_in_class = groups.get(node, [])
        n_props = len(props_in_class)

        hover_lines = [
            f"<b>{class_str}</b>",
            f"Propositions: {n_props}",
            f"In-degree: {graph.in_degree(node)}",
            f"Out-degree: {graph.out_degree(node)}"
        ]
        node_text.append("<br>".join(hover_lines))

        # Size based on number of propositions
        size = 15 + n_props * 3
        node_size.append(size)

    return go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_size,
            color=color,
            line=dict(width=2, color='white')
        ),
        name=f'{class_type}-classes'
    )


def create_mini_proposition_figure(verifier_states, falsifier_states, space,
                                   width=100, height=100, show_labels=True):
    """
    Create a compact Hasse diagram figure showing verifiers and falsifiers.

    Args:
        verifier_states: set of State objects (or state name strings) that are verifiers
        falsifier_states: set of State objects (or state name strings) that are falsifiers
        space: StateSpace object
        width: figure width in pixels
        height: figure height in pixels
        show_labels: whether to show state name labels on nodes

    Returns:
        plotly.graph_objects.Figure
    """
    # Build the state space graph
    edges = space.hasse_edges()
    G = nx.DiGraph()
    G.add_nodes_from(space.all_states())
    G.add_edges_from(edges)

    # Compute layout
    pos = compute_layout(G, 'hierarchical')

    # Normalize state names to strings for comparison
    def to_str(s):
        return s.to_string() if hasattr(s, 'to_string') else str(s)

    v_names = {to_str(s) for s in verifier_states} if verifier_states else set()
    f_names = {to_str(s) for s in falsifier_states} if falsifier_states else set()

    # Create edge trace
    edge_x, edge_y = [], []
    for e in edges:
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Determine node colors
    node_names = list(G.nodes())
    node_x = [pos[n][0] for n in node_names]
    node_y = [pos[n][1] for n in node_names]
    node_colors = []
    for name in node_names:
        in_v = name in v_names
        in_f = name in f_names
        if in_v and in_f:
            node_colors.append('gold')
        elif in_v:
            node_colors.append('limegreen')
        elif in_f:
            node_colors.append('tomato')
        else:
            node_colors.append('lightgray')

    # Create figure
    fig = go.Figure()

    # Edge trace
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1, color='darkgray'),
        hoverinfo='none'
    ))

    # Node trace
    node_size = 12 if len(node_names) <= 7 else (10 if len(node_names) <= 15 else 8)
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text' if show_labels else 'markers',
        marker=dict(
            size=node_size,
            color=node_colors,
            line=dict(width=1, color='darkgray')
        ),
        text=node_names if show_labels else None,
        textposition='middle center',
        textfont=dict(size=7, color='black'),
        hoverinfo='none'
    ))

    # Compact layout
    fig.update_layout(
        showlegend=False,
        margin=dict(l=2, r=2, t=2, b=2),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=width,
        height=height
    )

    return fig


def create_proposition_mini_figure(prop, space, width=100, height=100, show_labels=True):
    """
    Create a mini Hasse diagram for a BilateralProposition.

    Args:
        prop: BilateralProposition object
        space: StateSpace object
        width: figure width in pixels
        height: figure height in pixels
        show_labels: whether to show state name labels

    Returns:
        plotly.graph_objects.Figure
    """
    from truthmaker_core import all_states_in_regular_set

    # Get all states in the verifier and falsifier sets
    if prop.verifiers.is_null:
        v_states = set()
    else:
        v_states = all_states_in_regular_set(prop.verifiers)

    if prop.falsifiers.is_null:
        f_states = set()
    else:
        f_states = all_states_in_regular_set(prop.falsifiers)

    return create_mini_proposition_figure(v_states, f_states, space, width, height, show_labels)


def create_L_class_mini_figure(L_class, space, width=100, height=100, show_labels=True):
    """
    Create a mini Hasse diagram for an L-equivalence class.
    Shows upward closure of the antichains.

    Args:
        L_class: tuple of (verifier_antichain, falsifier_antichain) as frozensets of States
        space: StateSpace object
        width, height: figure dimensions
        show_labels: whether to show state labels

    Returns:
        plotly.graph_objects.Figure
    """
    v_antichain, f_antichain = L_class

    # Compute upward closure of each antichain
    def upward_closure(antichain):
        if not antichain:
            return set()
        result = set()
        all_states = space.all_states()
        for s in antichain:
            s_name = s.to_string() if hasattr(s, 'to_string') else str(s)
            for t in all_states:
                if space.leq(s_name, t):
                    result.add(t)
        return result

    v_states = upward_closure(v_antichain)
    f_states = upward_closure(f_antichain)

    return create_mini_proposition_figure(v_states, f_states, space, width, height, show_labels)


def create_M_class_mini_figure(M_class, space, width=100, height=100, show_labels=True):
    """
    Create a mini Hasse diagram for an M-equivalence class.
    Shows downward closure of the top states.

    Args:
        M_class: tuple of (verifier_top, falsifier_top) where each is a State or None
        space: StateSpace object
        width, height: figure dimensions
        show_labels: whether to show state labels

    Returns:
        plotly.graph_objects.Figure
    """
    v_top, f_top = M_class

    # Compute downward closure of each top state
    def downward_closure(top_state):
        if top_state is None:
            return set()
        top_name = top_state.to_string() if hasattr(top_state, 'to_string') else str(top_state)
        result = set()
        for s in space.all_states():
            if space.leq(s, top_name):
                result.add(s)
        return result

    v_states = downward_closure(v_top)
    f_states = downward_closure(f_top)

    return create_mini_proposition_figure(v_states, f_states, space, width, height, show_labels)

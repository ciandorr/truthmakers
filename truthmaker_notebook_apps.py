"""
truthmaker_notebook_apps.py

Dash/Plotly interactive applications for truthmaker semantics visualization.
This module provides ready-to-use apps for Jupyter notebooks.

Main components:
- create_proposition_builder_app(): Interactive proposition builder with closure operations
- create_dual_hasse_app(): Interactive dual Hasse diagram with L-classes and M-classes
"""

from dash import Dash, dcc, html, Input, Output, State as DashState, ALL, ctx
import plotly.graph_objects as go
import networkx as nx

from truthmaker_core import (
    State as TruthmakerState,
    RegularSet, BilateralProposition, PropositionSet,
    vee, singleton_regular_set, all_states_in_regular_set,
    get_state_space, stateset_to_string,
    conjoin, disjoin, negate,
    build_L_lattice, build_M_semilattice,
    group_by_L_class, group_by_M_class,
    get_L_class, get_M_class,
    L_class_to_string, M_class_to_string,
    has_null_m_class,
    parse_propositions
)

from truthmaker_visualization import (
    compute_layout, create_mini_proposition_figure,
    create_L_class_mini_figure, create_M_class_mini_figure
)


# =============================================================================
# Serialization helpers for Dash stores (convert to/from JSON-safe formats)
# =============================================================================

def serialize_regular_set(rs):
    """Serialize RegularSet to JSON-safe dict."""
    if rs is None or rs.is_null:
        return None
    return {
        'antichain': [s.to_string() for s in rs.antichain],
        'top': rs.top_state.to_string()
    }


def deserialize_regular_set(data):
    """Deserialize RegularSet from JSON-safe dict."""
    if data is None:
        return RegularSet.null()
    antichain = frozenset(TruthmakerState.from_string(s) for s in data['antichain'])
    top = TruthmakerState.from_string(data['top'])
    return RegularSet(antichain, top)


def serialize_L_class(L_class):
    """Serialize L-class (pair of antichains) to JSON-safe format."""
    v_antichain, f_antichain = L_class
    return [
        [s.to_string() for s in v_antichain],
        [s.to_string() for s in f_antichain]
    ]


def deserialize_L_class(data):
    """Deserialize L-class from JSON-safe format."""
    v_list, f_list = data
    v_antichain = frozenset(TruthmakerState.from_string(s) for s in v_list)
    f_antichain = frozenset(TruthmakerState.from_string(s) for s in f_list)
    return (v_antichain, f_antichain)


def serialize_M_class(M_class):
    """Serialize M-class (pair of top states) to JSON-safe format."""
    v_top, f_top = M_class
    return [
        v_top.to_string() if v_top is not None else None,
        f_top.to_string() if f_top is not None else None
    ]


def deserialize_M_class(data):
    """Deserialize M-class from JSON-safe format."""
    v_str, f_str = data
    v_top = TruthmakerState.from_string(v_str) if v_str is not None else None
    f_top = TruthmakerState.from_string(f_str) if f_str is not None else None
    return (v_top, f_top)


def bilateral_to_prop_dict(bp):
    """Convert BilateralProposition to storage dict format."""
    return {
        'verifiers': serialize_regular_set(bp.verifiers),
        'falsifiers': serialize_regular_set(bp.falsifiers),
        'string': bp.to_string().strip('()')
    }


def prop_dict_to_bilateral(prop_dict):
    """Convert storage dict to BilateralProposition."""
    v = deserialize_regular_set(prop_dict.get('verifiers'))
    f = deserialize_regular_set(prop_dict.get('falsifiers'))
    return BilateralProposition(v, f)


def regular_set_to_string(rs):
    """Convert RegularSet to display string."""
    if rs is None or rs.is_null:
        return "0"
    return f"{stateset_to_string(rs.antichain)};{rs.top_state.to_string()}"


def proposition_to_string(verifiers, falsifiers):
    """Convert verifier/falsifier pair to display string."""
    return f"{regular_set_to_string(verifiers)}|{regular_set_to_string(falsifiers)}"


# =============================================================================
# Proposition list helpers
# =============================================================================

def prop_list_to_proposition_set(prop_list):
    """
    Convert a PROPOSITION_LIST (list of dicts) to a PropositionSet.
    Removes duplicates and empty propositions.
    """
    if not prop_list:
        return None
    bilaterals = []
    seen = set()
    for p in prop_list:
        if p.get('string') != '0|0':
            bp = prop_dict_to_bilateral(p)
            key = bp.to_string()
            if key not in seen:
                bilaterals.append(bp)
                seen.add(key)
    return PropositionSet(bilaterals) if bilaterals else None


def generate_default_propositions():
    """
    Generate default mutually exclusive atom propositions for the current state space.

    For each atom a, creates proposition:
      (a;a | b1,...,bn; fuse(b1,...,bn))
    where b1,...,bn are all other atoms.

    Returns:
        list of prop_dict objects, or empty list if generation fails
    """
    space = get_state_space()
    if space is None:
        return []

    # Extract atoms (states with no proper parts except possibly bottom)
    states = space.all_states()

    # Find bottom states (≤ every other state)
    bottom_states = [s for s in states if all(space.leq(s, t) for t in states)]

    # Find atoms: non-bottom states whose only proper parts are bottom states
    atoms = []
    for s in states:
        if s in bottom_states:
            continue
        proper_parts = [t for t in states if space.leq(t, s) and t != s]
        non_bottom_parts = [t for t in proper_parts if t not in bottom_states]
        if not non_bottom_parts:
            atoms.append(s)

    atoms = sorted(atoms)

    if len(atoms) < 2:
        return []  # Need at least 2 atoms

    prop_dicts = []
    for a in atoms:
        other_atoms = [b for b in atoms if b != a]

        # Compute fusion of all other atoms
        f_top = other_atoms[0]
        for b in other_atoms[1:]:
            f_top = space.join(f_top, b)

        # Format: "a;a|b1,b2,...,bn;f_top"
        f_antichain_str = ','.join(other_atoms)
        prop_str = f"{a};{a}|{f_antichain_str};{f_top}"

        try:
            parsed = parse_propositions([prop_str])
            if parsed:
                prop_dicts.append(bilateral_to_prop_dict(parsed[0]))
        except Exception:
            pass  # Skip if parsing fails

    # Add bottom state proposition if exists
    for b in bottom_states:
        try:
            parsed = parse_propositions([f"{b};{b}|0"])
            if parsed:
                prop_dicts.append(bilateral_to_prop_dict(parsed[0]))
        except Exception:
            pass

    return prop_dicts


def parse_text_to_propositions(text):
    """
    Parse text input into proposition dicts.

    Accepts one proposition per line in format: antichain;top|antichain;top
    States in antichains can be comma or space separated.

    Returns:
        tuple of (list of prop_dicts, error_message or None)
    """
    prop_strings = [s.strip() for s in text.strip().split('\n') if s.strip()]
    if not prop_strings:
        return [], "No propositions entered"

    # Convert comma-separated to space-separated for parser
    converted = []
    for s in prop_strings:
        parts = s.split('|')
        converted_parts = []
        for part in parts:
            subparts = part.split(';')
            if len(subparts) >= 1:
                subparts[0] = subparts[0].replace(',', ' ')
            converted_parts.append(';'.join(subparts))
        converted.append('|'.join(converted_parts))

    try:
        parsed = parse_propositions(converted)
        return [bilateral_to_prop_dict(bp) for bp in parsed], None
    except Exception as e:
        return [], str(e)


# =============================================================================
# Proposition Builder App (Visual Version)
# =============================================================================

def create_proposition_builder_app(proposition_list, on_update=None):
    """
    Create interactive proposition builder Dash app with visual mini-figures.

    Features:
    - Visual grid of proposition mini-figures (click to select/multi-select)
    - Click states on main diagram to add to selected proposition
    - Single operations: NOT, AND, OR on selected propositions
    - Closure operations: close under NOT, AND, OR, or all

    Args:
        proposition_list: List of proposition dicts (will be modified in place)
        on_update: Optional callback function called when list changes

    Returns:
        Dash app ready to run with app.run(jupyter_mode='inline', ...)
    """
    space = get_state_space()

    # Build Hasse diagram structure
    edges = space.hasse_edges()
    G = nx.DiGraph()
    G.add_nodes_from(space.all_states())
    G.add_edges_from(edges)
    pos = compute_layout(G, 'hierarchical')

    # Prepare edge coordinates (static)
    edge_x, edge_y = [], []
    for e in edges:
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Determine mini-figure size based on state space
    n_states = len(space.all_states())
    mini_size = 90 if n_states <= 7 else (80 if n_states <= 15 else 70)

    def create_main_figure(verifiers, falsifiers):
        """Create the main state space figure for editing."""
        verifier_states = all_states_in_regular_set(verifiers) if verifiers and not verifiers.is_null else set()
        falsifier_states = all_states_in_regular_set(falsifiers) if falsifiers and not falsifiers.is_null else set()

        node_colors = []
        node_names = list(G.nodes())
        for name in node_names:
            state = TruthmakerState(name)
            in_v = state in verifier_states
            in_f = state in falsifier_states
            if in_v and in_f:
                node_colors.append('gold')
            elif in_v:
                node_colors.append('limegreen')
            elif in_f:
                node_colors.append('tomato')
            else:
                node_colors.append('lightgray')

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(width=1.5, color='darkgray'), hoverinfo='none'
        ))
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in node_names],
            y=[pos[n][1] for n in node_names],
            mode='markers+text',
            marker=dict(size=35, color=node_colors, line=dict(width=2, color='darkgray')),
            text=node_names, textposition='middle center',
            textfont=dict(size=12, color='black'),
            hoverinfo='text',
            hovertext=[f"State: {n}<br>Click to add" for n in node_names],
            customdata=node_names
        ))
        fig.update_layout(
            showlegend=False, hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
            plot_bgcolor='white', width=350, height=280,
            title=dict(text='Click states to add to first selected', font=dict(size=10))
        )
        return fig

    def get_prop_states(prop_dict):
        """Get verifier and falsifier state sets from prop dict."""
        v = deserialize_regular_set(prop_dict.get('verifiers'))
        f = deserialize_regular_set(prop_dict.get('falsifiers'))
        v_states = all_states_in_regular_set(v) if v and not v.is_null else set()
        f_states = all_states_in_regular_set(f) if f and not f.is_null else set()
        return v_states, f_states

    def create_prop_grid(props, selected_indices):
        """Create a grid of clickable mini-figures."""
        if not props:
            return html.Div("No propositions. Click 'Add New' to start.",
                           style={'color': 'gray', 'fontStyle': 'italic', 'padding': '10px', 'textAlign': 'center'})

        items = []
        for i, prop in enumerate(props):
            is_selected = i in selected_indices
            is_primary = selected_indices and i == selected_indices[0]  # Index 0 in list is primary

            v_states, f_states = get_prop_states(prop)
            mini_fig = create_mini_proposition_figure(v_states, f_states, space,
                                                       width=mini_size, height=mini_size,
                                                       show_labels=False)

            # Border color based on selection state
            if is_primary:
                border = '3px solid #ff6600'  # Orange for primary (being edited)
            elif is_selected:
                border = '3px solid #ffc107'  # Yellow for secondary selection
            else:
                border = '1px solid #ccc'

            item = html.Div([
                dcc.Graph(
                    figure=mini_fig,
                    config={'displayModeBar': False, 'staticPlot': True},
                    style={'width': f'{mini_size}px', 'height': f'{mini_size}px'}
                ),
                html.Div(f"{i+1}", style={
                    'textAlign': 'center', 'fontSize': '9px', 'color': '#666',
                    'marginTop': '-5px'
                })
            ],
                id={'type': 'mini-prop', 'index': i},
                n_clicks=0,
                style={
                    'display': 'inline-block', 'margin': '3px', 'padding': '2px',
                    'border': border, 'borderRadius': '5px', 'cursor': 'pointer',
                    'backgroundColor': '#fffef0' if is_selected else 'white',
                    'verticalAlign': 'top'
                }
            )
            items.append(item)

        return html.Div(items, style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'flex-start'})

    # Initialize with current list
    initial_props = list(proposition_list)
    initial_selected = [0] if initial_props else []  # List: [primary, secondary]

    app = Dash(__name__)

    app.layout = html.Div([
        html.H4("Visual Proposition Builder", style={'textAlign': 'center', 'marginBottom': '5px'}),

        html.Div([
            # Left: Main graph and mode toggle
            html.Div([
                html.Div([
                    html.Label("Mode: ", style={'fontWeight': 'bold', 'marginRight': '5px', 'fontSize': '11px'}),
                    dcc.RadioItems(
                        id='mode-toggle',
                        options=[
                            {'label': ' Verifiers (green)', 'value': 'verifiers'},
                            {'label': ' Falsifiers (red)', 'value': 'falsifiers'}
                        ],
                        value='verifiers', inline=True,
                        style={'display': 'inline-block', 'fontSize': '11px'}
                    ),
                ], style={'marginBottom': '3px'}),
                dcc.Graph(
                    id='builder-graph',
                    figure=create_main_figure(
                        deserialize_regular_set(initial_props[0].get('verifiers')) if initial_props else None,
                        deserialize_regular_set(initial_props[0].get('falsifiers')) if initial_props else None
                    ),
                    config={'displayModeBar': False}
                ),
                html.Div([
                    html.Button('Clear First', id='clear-btn', n_clicks=0,
                               style={'padding': '3px 8px', 'fontSize': '10px', 'marginRight': '5px'}),
                    html.Button('Select All', id='select-all-btn', n_clicks=0,
                               style={'padding': '3px 8px', 'fontSize': '10px', 'marginRight': '5px'}),
                    html.Button('Select None', id='select-none-btn', n_clicks=0,
                               style={'padding': '3px 8px', 'fontSize': '10px'}),
                ], style={'marginTop': '3px'}),
            ], style={'width': '38%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            # Right: Controls and proposition grid
            html.Div([
                # Row 0: Auto-Init and Add from text buttons
                html.Div([
                    html.Button('Auto-Init', id='auto-init-btn', n_clicks=0,
                               title='Generate default atom propositions',
                               style={'backgroundColor': '#ff9800', 'color': 'white',
                                      'padding': '3px 8px', 'marginRight': '4px', 'fontSize': '10px'}),
                    html.Button('Add from text', id='add-text-btn', n_clicks=0,
                               title='Parse propositions from text',
                               style={'backgroundColor': '#2196F3', 'color': 'white',
                                      'padding': '3px 8px', 'marginRight': '10px', 'fontSize': '10px'}),
                    html.Button('Add New', id='add-new-btn', n_clicks=0,
                               style={'backgroundColor': '#4CAF50', 'color': 'white',
                                      'padding': '3px 8px', 'marginRight': '4px', 'fontSize': '10px'}),
                    html.Button('Delete Sel', id='delete-btn', n_clicks=0,
                               style={'backgroundColor': '#dc3545', 'color': 'white',
                                      'padding': '3px 8px', 'marginRight': '4px', 'fontSize': '10px'}),
                    html.Button('Delete All', id='delete-all-btn', n_clicks=0,
                               style={'backgroundColor': '#6c757d', 'color': 'white',
                                      'padding': '3px 8px', 'fontSize': '10px'}),
                ], style={'marginBottom': '4px'}),

                # Row 1: Single operations
                html.Div([
                    html.Label("Apply to selected: ", style={'fontWeight': 'bold', 'fontSize': '10px', 'marginRight': '4px'}),
                    html.Button('NOT', id='single-neg-btn', n_clicks=0, title='Negate selected (adds result)',
                               style={'padding': '2px 6px', 'marginRight': '3px', 'fontSize': '10px',
                                      'backgroundColor': '#17a2b8', 'color': 'white'}),
                    html.Button('AND', id='single-conj-btn', n_clicks=0, title='Conjoin 2 selected (adds result)',
                               style={'padding': '2px 6px', 'marginRight': '3px', 'fontSize': '10px',
                                      'backgroundColor': '#17a2b8', 'color': 'white'}),
                    html.Button('OR', id='single-disj-btn', n_clicks=0, title='Disjoin 2 selected (adds result)',
                               style={'padding': '2px 6px', 'fontSize': '10px',
                                      'backgroundColor': '#17a2b8', 'color': 'white'}),
                ], style={'marginBottom': '4px'}),

                # Row 2: Closure operations
                html.Div([
                    html.Label("Close under: ", style={'fontWeight': 'bold', 'fontSize': '10px', 'marginRight': '4px'}),
                    html.Button('NOT', id='close-neg-btn', n_clicks=0,
                               style={'padding': '2px 6px', 'marginRight': '3px', 'fontSize': '10px'}),
                    html.Button('AND', id='close-conj-btn', n_clicks=0,
                               style={'padding': '2px 6px', 'marginRight': '3px', 'fontSize': '10px'}),
                    html.Button('OR', id='close-disj-btn', n_clicks=0,
                               style={'padding': '2px 6px', 'marginRight': '3px', 'fontSize': '10px'}),
                    html.Button('All', id='close-all-btn', n_clicks=0,
                               style={'padding': '2px 6px', 'backgroundColor': '#9C27B0',
                                      'color': 'white', 'fontSize': '10px'}),
                ], style={'marginBottom': '4px', 'padding': '4px', 'backgroundColor': '#f0f0f0', 'borderRadius': '4px'}),

                # Status and count
                html.Div([
                    html.Span(id='prop-count', children=f"{len(initial_props)} propositions",
                             style={'fontSize': '11px', 'fontWeight': 'bold'}),
                    html.Span(" | ", style={'color': '#ccc'}),
                    html.Span(id='selection-info', children=f"{len(initial_selected)} selected",
                             style={'fontSize': '11px', 'color': '#666'}),
                    html.Span(" | ", style={'color': '#ccc'}),
                    html.Span(id='status-msg', style={'fontSize': '11px', 'color': '#28a745'}),
                ], style={'marginBottom': '4px'}),

                # Proposition grid (scrollable)
                html.Div(
                    id='prop-grid-container',
                    children=create_prop_grid(initial_props, initial_selected),
                    style={
                        'height': '220px', 'overflowY': 'auto', 'border': '1px solid #ccc',
                        'borderRadius': '5px', 'padding': '5px', 'backgroundColor': '#fafafa'
                    }
                ),
            ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'}),
        ]),

        # Text input modal
        html.Div(
            id='text-modal',
            children=[
                html.Div([
                    html.H5("Add Propositions from Text", style={'marginBottom': '10px'}),
                    html.P("Enter one proposition per line. Format: antichain;top|antichain;top",
                          style={'fontSize': '11px', 'color': '#666', 'marginBottom': '5px'}),
                    html.P("States in antichains can be comma or space separated.",
                          style={'fontSize': '11px', 'color': '#666', 'marginBottom': '10px'}),
                    dcc.Textarea(
                        id='text-input-area',
                        placeholder='r;r|g,b;gb\ng;g|r,b;rb\nb;b|r,g;rg',
                        style={'width': '100%', 'height': '120px', 'fontFamily': 'monospace', 'fontSize': '12px'}
                    ),
                    html.Div(id='text-parse-error', style={'color': 'red', 'fontSize': '11px', 'marginTop': '5px'}),
                    html.Div([
                        html.Button('Add', id='text-add-btn', n_clicks=0,
                                   style={'backgroundColor': '#4CAF50', 'color': 'white',
                                          'padding': '5px 15px', 'marginRight': '10px'}),
                        html.Button('Cancel', id='text-cancel-btn', n_clicks=0,
                                   style={'padding': '5px 15px'}),
                    ], style={'marginTop': '10px', 'textAlign': 'right'})
                ], style={
                    'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px',
                    'boxShadow': '0 4px 20px rgba(0,0,0,0.3)', 'maxWidth': '500px', 'margin': 'auto'
                })
            ],
            style={
                'display': 'none', 'position': 'fixed', 'top': '0', 'left': '0',
                'width': '100%', 'height': '100%', 'backgroundColor': 'rgba(0,0,0,0.5)',
                'zIndex': '1000', 'paddingTop': '100px'
            }
        ),

        # Stores
        dcc.Store(id='props-store', data=initial_props),
        dcc.Store(id='selected-set-store', data=list(initial_selected)),
        dcc.Store(id='modal-open', data=False),
    ], style={'padding': '8px', 'maxWidth': '950px'})

    @app.callback(
        [Output('builder-graph', 'figure'), Output('props-store', 'data'),
         Output('selected-set-store', 'data'), Output('prop-grid-container', 'children'),
         Output('prop-count', 'children'), Output('selection-info', 'children'),
         Output('status-msg', 'children'), Output('text-modal', 'style'),
         Output('text-input-area', 'value'), Output('text-parse-error', 'children'),
         Output('mode-toggle', 'value')],
        [Input('builder-graph', 'clickData'), Input('clear-btn', 'n_clicks'),
         Input('add-new-btn', 'n_clicks'), Input('delete-btn', 'n_clicks'),
         Input('delete-all-btn', 'n_clicks'), Input('select-all-btn', 'n_clicks'),
         Input('select-none-btn', 'n_clicks'),
         Input('single-neg-btn', 'n_clicks'), Input('single-conj-btn', 'n_clicks'),
         Input('single-disj-btn', 'n_clicks'),
         Input('close-neg-btn', 'n_clicks'), Input('close-conj-btn', 'n_clicks'),
         Input('close-disj-btn', 'n_clicks'), Input('close-all-btn', 'n_clicks'),
         Input({'type': 'mini-prop', 'index': ALL}, 'n_clicks'),
         Input('auto-init-btn', 'n_clicks'), Input('add-text-btn', 'n_clicks'),
         Input('text-add-btn', 'n_clicks'), Input('text-cancel-btn', 'n_clicks')],
        [DashState('mode-toggle', 'value'), DashState('props-store', 'data'),
         DashState('selected-set-store', 'data'), DashState('text-input-area', 'value')]
    )
    def update_builder(clickData, clear_clicks, add_clicks, delete_clicks, delete_all_clicks,
                       select_all_clicks, select_none_clicks,
                       single_neg, single_conj, single_disj,
                       close_neg, close_conj, close_disj, close_all, mini_clicks,
                       auto_init_clicks, add_text_clicks, text_add_clicks, text_cancel_clicks,
                       mode, props, selected_list, text_input):
        triggered = ctx.triggered_id
        status = ""
        # selected is a LIST where index 0 = primary (orange), index 1 = secondary (yellow)
        selected = list(selected_list) if selected_list else []
        new_mode = mode  # Track mode changes
        parse_error = ""

        # Modal styles
        modal_hidden = {
            'display': 'none', 'position': 'fixed', 'top': '0', 'left': '0',
            'width': '100%', 'height': '100%', 'backgroundColor': 'rgba(0,0,0,0.5)',
            'zIndex': '1000', 'paddingTop': '100px'
        }
        modal_visible = {
            'display': 'block', 'position': 'fixed', 'top': '0', 'left': '0',
            'width': '100%', 'height': '100%', 'backgroundColor': 'rgba(0,0,0,0.5)',
            'zIndex': '1000', 'paddingTop': '100px'
        }
        modal_style = modal_hidden
        text_area_value = text_input or ""

        # Auto-Init: generate default atom propositions
        if triggered == 'auto-init-btn':
            new_props = generate_default_propositions()
            if new_props:
                props.extend(new_props)
                status = f"Added {len(new_props)} default propositions"
                selected = [len(props) - 1] if props else []
            else:
                status = "Could not generate defaults (need >= 2 atoms)"

        # Open text input modal
        elif triggered == 'add-text-btn':
            modal_style = modal_visible
            text_area_value = ""  # Clear the text area

        # Add from text modal
        elif triggered == 'text-add-btn' and text_input:
            new_props, error = parse_text_to_propositions(text_input)
            if error:
                parse_error = f"Error: {error}"
                modal_style = modal_visible  # Keep modal open
            elif new_props:
                props.extend(new_props)
                status = f"Added {len(new_props)} propositions from text"
                selected = [len(props) - 1] if props else []
                text_area_value = ""  # Clear on success
            else:
                parse_error = "No valid propositions found"
                modal_style = modal_visible

        # Cancel text modal
        elif triggered == 'text-cancel-btn':
            text_area_value = ""

        # Add New
        elif triggered == 'add-new-btn':
            props.append({'verifiers': None, 'falsifiers': None, 'string': '0|0'})
            selected = [len(props) - 1]  # Select the new one as primary
            new_mode = 'verifiers'  # Switch to verifier mode for new proposition

        # Delete Selected
        elif triggered == 'delete-btn' and selected and props:
            # Delete in reverse order to maintain indices
            for idx in sorted(selected, reverse=True):
                if 0 <= idx < len(props):
                    props.pop(idx)
            selected = [0] if props else []

        # Delete All
        elif triggered == 'delete-all-btn':
            props = []
            selected = []

        # Select All (first becomes primary, rest secondary - but we only allow 2 max)
        elif triggered == 'select-all-btn' and props:
            selected = [0, 1] if len(props) >= 2 else ([0] if props else [])

        # Select None
        elif triggered == 'select-none-btn':
            selected = []

        # Click mini-prop: clicked becomes primary, old primary becomes secondary, old secondary cleared
        # selected is a list: [primary, secondary] where primary is index 0
        elif isinstance(triggered, dict) and triggered.get('type') == 'mini-prop':
            idx = triggered['index']
            if idx in selected:
                # Clicking already-selected item
                if len(selected) == 2 and selected[1] == idx:
                    # Clicking secondary: make it primary, old primary becomes secondary
                    selected = [idx, selected[0]]
                # If clicking primary, no change needed
            else:
                # Clicking new item: it becomes primary, old primary becomes secondary
                if selected:
                    old_primary = selected[0]
                    selected = [idx, old_primary]
                else:
                    selected = [idx]

        # Clear first selected (primary)
        elif triggered == 'clear-btn' and selected and props:
            first = selected[0]
            if 0 <= first < len(props):
                props[first] = {'verifiers': None, 'falsifiers': None, 'string': '0|0'}

        # Graph click - add state to primary selected proposition
        elif triggered == 'builder-graph' and clickData and selected and props:
            first = selected[0]
            if 0 <= first < len(props):
                point = clickData['points'][0]
                if 'customdata' in point:
                    clicked_state = TruthmakerState(point['customdata'])
                    singleton = singleton_regular_set(clicked_state)
                    v = deserialize_regular_set(props[first].get('verifiers'))
                    f = deserialize_regular_set(props[first].get('falsifiers'))
                    if mode == 'verifiers':
                        v = vee(v, singleton)
                    else:
                        f = vee(f, singleton)
                    props[first] = {
                        'verifiers': serialize_regular_set(v),
                        'falsifiers': serialize_regular_set(f),
                        'string': proposition_to_string(v, f)
                    }

        # Single NOT - negate primary selected, add result
        elif triggered == 'single-neg-btn' and selected and props:
            idx = selected[0]  # Only negate primary
            if 0 <= idx < len(props) and props[idx].get('string') != '0|0':
                bp = prop_dict_to_bilateral(props[idx])
                neg_bp = negate(bp)
                props.append(bilateral_to_prop_dict(neg_bp))
                status = "Added negation"
                selected = [len(props) - 1]  # Select new as primary

        # Single AND - conjoin primary and secondary (exactly 2 selected)
        elif triggered == 'single-conj-btn' and len(selected) == 2 and props:
            idx1, idx2 = selected[0], selected[1]
            if all(0 <= i < len(props) and props[i].get('string') != '0|0' for i in [idx1, idx2]):
                bp1 = prop_dict_to_bilateral(props[idx1])
                bp2 = prop_dict_to_bilateral(props[idx2])
                conj_bp = conjoin(bp1, bp2)
                props.append(bilateral_to_prop_dict(conj_bp))
                status = "Added conjunction"
                selected = [len(props) - 1]  # Select new as primary
            else:
                status = "Select 2 non-empty propositions"

        # Single OR - disjoin primary and secondary (exactly 2 selected)
        elif triggered == 'single-disj-btn' and len(selected) == 2 and props:
            idx1, idx2 = selected[0], selected[1]
            if all(0 <= i < len(props) and props[i].get('string') != '0|0' for i in [idx1, idx2]):
                bp1 = prop_dict_to_bilateral(props[idx1])
                bp2 = prop_dict_to_bilateral(props[idx2])
                disj_bp = disjoin(bp1, bp2)
                props.append(bilateral_to_prop_dict(disj_bp))
                status = "Added disjunction"
                selected = [len(props) - 1]  # Select new as primary
            else:
                status = "Select 2 non-empty propositions"

        # Closure operations (on all props)
        elif triggered in ['close-neg-btn', 'close-conj-btn', 'close-disj-btn', 'close-all-btn'] and props:
            bilaterals = []
            seen = set()
            for p in props:
                if p.get('string') != '0|0':
                    bp = prop_dict_to_bilateral(p)
                    key = bp.to_string()
                    if key not in seen:
                        bilaterals.append(bp)
                        seen.add(key)

            if bilaterals:
                prop_set = PropositionSet(bilaterals)
                initial_count = len(prop_set)

                if triggered == 'close-neg-btn':
                    prop_set.negclose()
                    status = f"NOT closure: {initial_count} -> {len(prop_set)}"
                elif triggered == 'close-conj-btn':
                    prop_set.conjclose()
                    status = f"AND closure: {initial_count} -> {len(prop_set)}"
                elif triggered == 'close-disj-btn':
                    prop_set.disjclose()
                    status = f"OR closure: {initial_count} -> {len(prop_set)}"
                elif triggered == 'close-all-btn':
                    prop_set.close()
                    status = f"Full closure: {initial_count} -> {len(prop_set)}"

                props = [bilateral_to_prop_dict(bp) for bp in prop_set]
                selected = [0] if props else []

        # Validate selected indices (keep only valid, maintain order)
        selected = [i for i in selected if 0 <= i < len(props)]

        # Update the external list in place
        proposition_list.clear()
        proposition_list.extend(props)
        if on_update:
            on_update(props)

        # Get primary selected proposition for main graph
        if selected and props:
            first = selected[0]  # Primary is index 0 in list
            v = deserialize_regular_set(props[first].get('verifiers'))
            f = deserialize_regular_set(props[first].get('falsifiers'))
        else:
            v, f = RegularSet.null(), RegularSet.null()

        fig = create_main_figure(v, f)
        grid = create_prop_grid(props, selected)
        count_str = f"{len(props)} propositions"
        sel_str = f"{len(selected)} selected"

        return fig, props, selected, grid, count_str, sel_str, status, modal_style, text_area_value, parse_error, new_mode

    return app


# =============================================================================
# Dual Hasse Diagram App
# =============================================================================

def create_dual_hasse_app(prop_set, m_ordering='A'):
    """
    Create interactive dual Hasse diagram Dash app.

    Shows L-equivalence class lattice and M-equivalence class semilattice
    side by side, with propositions as red lines connecting them.
    Hovering over nodes shows mini Hasse diagram previews.

    Args:
        prop_set: PropositionSet or list of BilateralPropositions
        m_ordering: 'A' or 'B' - ordering option for M-classes when nulls present
            Option A: null verifiers at bottom, null falsifiers at top
            Option B: null verifiers at top, null falsifiers at bottom

    Returns:
        Dash app ready to run with app.run(jupyter_mode='inline', ...)
    """
    props = prop_set.to_list() if isinstance(prop_set, PropositionSet) else list(prop_set)
    space = get_state_space()

    # Build full structures (no transitive reduction)
    L_lattice = build_L_lattice(props)
    M_semilattice = build_M_semilattice(props, ordering_option=m_ordering)

    # Compute transitive reductions for layout and default display
    L_lattice_tr = nx.transitive_reduction(L_lattice)
    M_semilattice_tr = nx.transitive_reduction(M_semilattice)

    L_groups = group_by_L_class(props)
    M_groups = group_by_M_class(props)

    # Compute layouts using TR
    L_pos_raw = compute_layout(L_lattice_tr, 'hierarchical')
    M_pos_raw = compute_layout(M_semilattice_tr, 'hierarchical')
    L_pos = {node: (x * 0.35, y) for node, (x, y) in L_pos_raw.items()}
    M_pos = {node: (0.65 + x * 0.35, y) for node, (x, y) in M_pos_raw.items()}

    def create_dynamic_edge_traces(full_graph, full_tr, pos, secondary_nodes=None):
        """Create edge traces with dynamic TR highlighting."""
        secondary_nodes = secondary_nodes or set()

        if secondary_nodes:
            subgraph = full_graph.subgraph(secondary_nodes)
            if subgraph.number_of_nodes() > 1 and subgraph.number_of_edges() > 0:
                highlighted_tr = nx.transitive_reduction(subgraph)
                highlighted_edges = set(highlighted_tr.edges())
            else:
                highlighted_edges = set()
        else:
            highlighted_edges = set(full_tr.edges())

        thin_x, thin_y, fat_x, fat_y = [], [], [], []
        for edge in full_graph.edges():
            if edge[0] not in pos or edge[1] not in pos:
                continue
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            if edge in highlighted_edges:
                fat_x.extend([x0, x1, None])
                fat_y.extend([y0, y1, None])
            else:
                thin_x.extend([x0, x1, None])
                thin_y.extend([y0, y1, None])

        traces = []
        if thin_x:
            traces.append(go.Scatter(
                x=thin_x, y=thin_y,
                line=dict(width=0.5, color='rgba(180,180,180,0.5)'),
                hoverinfo='none', mode='lines', name='Non-TR'
            ))
        if fat_x:
            traces.append(go.Scatter(
                x=fat_x, y=fat_y,
                line=dict(width=2, color='#333'),
                hoverinfo='none', mode='lines', name='TR'
            ))
        return traces

    def create_figure(primary_L=None, primary_M=None, secondary_L=None, secondary_M=None, selected_props=None):
        secondary_L = secondary_L or set()
        secondary_M = secondary_M or set()
        selected_props = selected_props or set()

        traces = []

        # Edges with dynamic TR
        traces.extend(create_dynamic_edge_traces(L_lattice, L_lattice_tr, L_pos, secondary_L))
        traces.extend(create_dynamic_edge_traces(M_semilattice, M_semilattice_tr, M_pos, secondary_M))

        # Proposition lines
        unsel_x, unsel_y, unsel_t = [], [], []
        sel_x, sel_y, sel_t = [], [], []
        for i, prop in enumerate(props):
            lc, mc = get_L_class(prop), get_M_class(prop)
            if lc not in L_pos or mc not in M_pos:
                continue
            x0, y0 = L_pos[lc]
            x1, y1 = M_pos[mc]
            t = prop.to_string()
            if i in selected_props:
                sel_x.extend([x0, x1, None])
                sel_y.extend([y0, y1, None])
                sel_t.extend([t, t, ''])
            else:
                unsel_x.extend([x0, x1, None])
                unsel_y.extend([y0, y1, None])
                unsel_t.extend([t, t, ''])

        if unsel_x:
            traces.append(go.Scatter(
                x=unsel_x, y=unsel_y,
                line=dict(width=0.5, color='rgba(255,0,0,0.15)'),
                hoverinfo='text', text=unsel_t, mode='lines'
            ))
        if sel_x:
            traces.append(go.Scatter(
                x=sel_x, y=sel_y,
                line=dict(width=1.5, color='rgba(255,0,0,0.8)'),
                hoverinfo='text', text=sel_t, mode='lines'
            ))

        # Nodes with percentile-based sizing
        min_size, max_size = 12, 28
        for graph, pos_dict, groups, primary, secondary, color, ctype in [
            (L_lattice, L_pos, L_groups, primary_L, secondary_L, 'lightblue', 'L'),
            (M_semilattice, M_pos, M_groups, primary_M, secondary_M, 'lightgreen', 'M')
        ]:
            nodes_list = list(graph.nodes())
            prop_counts = [len(groups.get(n, [])) for n in nodes_list]

            # Compute percentile ranks
            if len(prop_counts) > 1:
                sorted_counts = sorted(prop_counts)
                def get_pct(c):
                    positions = [i for i, x in enumerate(sorted_counts) if x == c]
                    avg_pos = sum(positions) / len(positions)
                    return avg_pos / (len(sorted_counts) - 1) if len(sorted_counts) > 1 else 0.5
                percentiles = [get_pct(c) for c in prop_counts]
            else:
                percentiles = [0.5] * len(prop_counts)

            nx_list, ny_list, ntext, ncolor, nsize, ncustom, nsymbol = [], [], [], [], [], [], []
            for i, node in enumerate(nodes_list):
                x, y = pos_dict[node]
                nx_list.append(x)
                ny_list.append(y)
                n_props = prop_counts[i]
                if ctype == 'L':
                    ntext.append(f"{L_class_to_string(node)}<br>Props: {n_props}")
                    ncustom.append(serialize_L_class(node))
                    nsymbol.append('circle')
                else:
                    ntext.append(f"{M_class_to_string(node)}<br>Props: {n_props}")
                    ncustom.append(serialize_M_class(node))
                    # M-classes with null verifiers or falsifiers are squares
                    v_top, f_top = node
                    nsymbol.append('square' if v_top is None or f_top is None else 'circle')

                nsize.append(min_size + percentiles[i] * (max_size - min_size))
                if node == primary:
                    ncolor.append('darkorange')
                elif node in secondary:
                    ncolor.append('gold')
                else:
                    ncolor.append(color)

            traces.append(go.Scatter(
                x=nx_list, y=ny_list, mode='markers', hoverinfo='text',
                text=ntext,
                marker=dict(size=nsize, color=ncolor, symbol=nsymbol, line=dict(width=2, color='white')),
                customdata=ncustom, name=f'{ctype}-classes'
            ))

        fig = go.Figure(data=traces)

        title = f"Dual Hasse Diagram ({len(props)} props, {len(L_groups)} L-classes, {len(M_groups)} M-classes)"

        fig.update_layout(
            title=title,
            showlegend=False, hovermode='closest',
            margin=dict(b=40, l=40, r=40, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.05]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.05]),
            plot_bgcolor='white', width=1000, height=700,
            annotations=[
                dict(x=0.175, y=1.05, xref='x', yref='paper',
                     text='<b>L-Classes (Lattice)</b>', showarrow=False, font=dict(size=14)),
                dict(x=0.825, y=1.05, xref='x', yref='paper',
                     text='<b>M-Classes (Semilattice)</b>', showarrow=False, font=dict(size=14))
            ]
        )
        return fig

    # Create empty preview figure
    def create_empty_preview():
        fig = go.Figure()
        fig.update_layout(
            margin=dict(l=2, r=2, t=2, b=2),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
            plot_bgcolor='#f8f8f8', paper_bgcolor='#f8f8f8',
            width=120, height=120,
            annotations=[dict(
                x=0.5, y=0.5, xref='paper', yref='paper',
                text='Hover over<br>a node', showarrow=False,
                font=dict(size=10, color='gray')
            )]
        )
        return fig

    # Create Dash app
    app = Dash(__name__)

    app.layout = html.Div([
        html.H3("Interactive Dual Hasse Diagram", style={'textAlign': 'center'}),
        html.Div([
            html.Button('Reset', id='reset-btn', n_clicks=0),
            html.Div(id='info', style={'marginLeft': '20px', 'display': 'inline-block'})
        ], style={'padding': '10px'}),

        # Main content with graph and preview panel (using flexbox)
        html.Div([
            # Graph (main area)
            html.Div([
                dcc.Graph(id='graph', figure=create_figure(), config={'displayModeBar': True})
            ], style={'flex': '0 0 auto'}),

            # Preview panel (right side, fixed position)
            html.Div([
                html.Div("Hover Preview", style={
                    'fontWeight': 'bold', 'fontSize': '11px', 'textAlign': 'center',
                    'marginBottom': '3px', 'color': '#666'
                }),
                html.Div(id='preview-label', style={
                    'fontSize': '9px', 'textAlign': 'center', 'marginBottom': '3px',
                    'color': '#333', 'fontFamily': 'monospace', 'minHeight': '14px'
                }),
                dcc.Graph(
                    id='preview-graph',
                    figure=create_empty_preview(),
                    config={'displayModeBar': False, 'staticPlot': True},
                    style={'width': '120px', 'height': '120px'}
                ),
                html.Div([
                    html.Div(style={'display': 'inline-block', 'width': '10px', 'height': '10px',
                                   'backgroundColor': 'limegreen', 'marginRight': '3px', 'verticalAlign': 'middle'}),
                    html.Span("V ", style={'fontSize': '9px'}),
                    html.Div(style={'display': 'inline-block', 'width': '10px', 'height': '10px',
                                   'backgroundColor': 'tomato', 'marginRight': '3px', 'marginLeft': '5px', 'verticalAlign': 'middle'}),
                    html.Span("F ", style={'fontSize': '9px'}),
                    html.Div(style={'display': 'inline-block', 'width': '10px', 'height': '10px',
                                   'backgroundColor': 'gold', 'marginRight': '3px', 'marginLeft': '5px', 'verticalAlign': 'middle'}),
                    html.Span("V&F", style={'fontSize': '9px'}),
                ], style={'textAlign': 'center', 'marginTop': '5px'})
            ], style={
                'flex': '0 0 auto', 'marginLeft': '10px', 'marginTop': '50px',
                'padding': '8px', 'backgroundColor': '#f8f8f8', 'borderRadius': '5px',
                'border': '1px solid #ddd', 'alignSelf': 'flex-start'
            })
        ], style={'display': 'flex', 'flexWrap': 'nowrap', 'alignItems': 'flex-start'}),

        dcc.Store(id='store-L', data={}),
        dcc.Store(id='store-M', data={}),
        dcc.Store(id='store-props', data=[])
    ])

    @app.callback(
        [Output('graph', 'figure'), Output('store-L', 'data'), Output('store-M', 'data'),
         Output('store-props', 'data'), Output('info', 'children')],
        [Input('graph', 'clickData'), Input('reset-btn', 'n_clicks')],
        [DashState('store-L', 'data'), DashState('store-M', 'data'), DashState('store-props', 'data')]
    )
    def update(clickData, reset, store_L, store_M, store_props):
        import dash
        primary_L, primary_M = None, None
        secondary_L, secondary_M = set(), set()
        selected_props = set(store_props or [])

        if store_L and 'primary' in store_L and store_L['primary']:
            primary_L = deserialize_L_class(store_L['primary'])
            secondary_M = set(deserialize_M_class(item) for item in store_L.get('secondary', []))
        if store_M and 'primary' in store_M and store_M['primary']:
            primary_M = deserialize_M_class(store_M['primary'])
            secondary_L = set(deserialize_L_class(item) for item in store_M.get('secondary', []))

        triggered = dash.callback_context
        if triggered.triggered and triggered.triggered[0]['prop_id'] == 'reset-btn.n_clicks':
            primary_L, primary_M = None, None
            secondary_L, secondary_M, selected_props = set(), set(), set()
            info = "Selection cleared"
        elif clickData and 'customdata' in clickData['points'][0]:
            primary_L, primary_M = None, None
            secondary_L, secondary_M, selected_props = set(), set(), set()

            cd = clickData['points'][0]['customdata']
            is_L = isinstance(cd[0], list)

            if is_L:
                primary_L = deserialize_L_class(cd)
                for i, prop in enumerate(props):
                    if get_L_class(prop) == primary_L:
                        selected_props.add(i)
                        secondary_M.add(get_M_class(prop))
                info = f"L: {L_class_to_string(primary_L)} ({len(selected_props)} props -> {len(secondary_M)} M-classes)"
            else:
                primary_M = deserialize_M_class(cd)
                for i, prop in enumerate(props):
                    if get_M_class(prop) == primary_M:
                        selected_props.add(i)
                        secondary_L.add(get_L_class(prop))
                info = f"M: {M_class_to_string(primary_M)} ({len(selected_props)} props -> {len(secondary_L)} L-classes)"
        else:
            info = "Click a node to explore"

        fig = create_figure(primary_L, primary_M, secondary_L, secondary_M, selected_props)
        store_L_new = {
            'primary': serialize_L_class(primary_L) if primary_L else None,
            'secondary': [serialize_M_class(m) for m in secondary_M]
        }
        store_M_new = {
            'primary': serialize_M_class(primary_M) if primary_M else None,
            'secondary': [serialize_L_class(l) for l in secondary_L]
        }
        return fig, store_L_new, store_M_new, list(selected_props), info

    @app.callback(
        [Output('preview-graph', 'figure'), Output('preview-label', 'children')],
        [Input('graph', 'hoverData')]
    )
    def update_preview(hoverData):
        """Update preview panel when hovering over L-class or M-class nodes."""
        if not hoverData or 'customdata' not in hoverData['points'][0]:
            return create_empty_preview(), ""

        cd = hoverData['points'][0]['customdata']
        is_L = isinstance(cd[0], list)

        if is_L:
            # L-class: show upward closure of antichains
            L_class = deserialize_L_class(cd)
            label = L_class_to_string(L_class)
            fig = create_L_class_mini_figure(L_class, space, width=120, height=120, show_labels=False)
        else:
            # M-class: show downward closure of top states
            M_class = deserialize_M_class(cd)
            label = M_class_to_string(M_class)
            fig = create_M_class_mini_figure(M_class, space, width=120, height=120, show_labels=False)

        return fig, label

    return app

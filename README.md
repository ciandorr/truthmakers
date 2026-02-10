# Truthmaker Semantics Visualization

An interactive Python toolkit for exploring Kit Fine's bilateral truthmaker semantics with modern Plotly/Dash visualizations in Jupyter notebooks.

## Overview

This project provides tools for working with bilateral propositions in truthmaker semantics:
- Define state spaces and atomic propositions with verifiers and falsifiers
- Generate closures under logical operations (AND, OR, NOT)
- Visualize propositions on Hasse diagrams
- Explore L-equivalence and M-equivalence class structures
- Interactive proposition builder with visual feedback

## Installation

```bash
pip install -r requirements.txt
```

Or install dependencies directly:

```bash
pip install networkx plotly dash
```

## Quick Start

1. **Open the notebook:**

```bash
jupyter notebook truthmaker_viz.ipynb
```

2. **Run the cells in order** to set up the state space, build propositions, and visualize.

3. **Use the Visual Proposition Builder** to create and manipulate propositions interactively.

## File Structure

```
truthmakers/
├── truthmaker_core.py          # Core data structures and operations
├── truthmaker_visualization.py # Visualization helper functions
├── truthmaker_notebook_apps.py # Dash apps for Jupyter (builder, dual Hasse)
├── truthmaker_viz.ipynb        # Interactive Jupyter notebook
├── test_truthmaker.py          # Unit tests
├── test_dual_hasse.py          # Tests for dual Hasse diagram
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Key Concepts

### States

A state is a possible way things might be. In a default state space, states are sets of atoms (basic ways things might be). For example, with atoms `r`, `g`, `b`:
- `r` is the state where only red is present
- `rg` is the state where red and green are both present
- `rgb` (or `■`) is the "top" state where all atoms are present

### Regular Sets

A regular set of states is a downward-closed set with a maximum element. It is encoded compactly as an (antichain, top_state) pair:
- **Antichain**: Minimal elements (no element is a part of another)
- **Top state**: Maximum element (all other elements are parts of it)

### Bilateral Propositions

A proposition has both verifiers (states that make it true) and falsifiers (states that make it false). Each is a regular set.

Format: `verifiers|falsifiers` where each side is `antichain;top`

**Example:** `r;r|g,b;gb`
- Verifiers: {r} with top r
- Falsifiers: {g, b} with top gb

### Operations

- **Conjunction** (AND): Combines verifiers via wedge (∧), falsifiers via vee (∨)
- **Disjunction** (OR): Combines verifiers via vee, falsifiers via wedge
- **Negation** (NOT): Swaps verifiers and falsifiers

### Equivalence Classes

- **L-class**: Propositions with the same antichains (regardless of top states)
- **M-class**: Propositions with the same top states (regardless of antichains)

The dual Hasse diagram visualizes L-classes as a lattice and M-classes as a semilattice, with propositions shown as lines connecting their L-class to their M-class.

## Using the Notebook

### 1. Set Up State Space

```python
from truthmaker_core import StateSpace, set_state_space

# Create a state space with 4 atoms
space = StateSpace.from_atom_string("rgby", include_empty=True)
set_state_space(space)
```

### 2. Build Propositions

Use the **Visual Proposition Builder** to:
- Click states on the Hasse diagram to add them as verifiers or falsifiers
- Use Auto-Init to generate default atom propositions
- Apply NOT, AND, OR operations to create new propositions
- Close under operations to generate the full Boolean structure

### 3. Visualize Dual Hasse Diagram

The **Interactive Dual Hasse Diagram** shows:
- L-equivalence classes (left) as a lattice
- M-equivalence classes (right) as a semilattice
- Propositions as red lines connecting their L-class to their M-class
- Hover preview showing which states are in each equivalence class

## API Reference

### Core Classes

```python
from truthmaker_core import (
    StateSpace, State, RegularSet, BilateralProposition, PropositionSet,
    parse_propositions, conjoin, disjoin, negate
)
```

### Visualization

```python
from truthmaker_notebook_apps import (
    create_proposition_builder_app,
    create_dual_hasse_app
)
```

## Running Tests

```bash
pytest test_truthmaker.py test_dual_hasse.py -v
```

## References

- Kit Fine, "A Theory of Truthmaker Content I: Conjunction, Disjunction and Negation" (2017)
- Kit Fine, "A Theory of Truthmaker Content II: Subject-matter, Common Content, Remainder and Ground" (2017)

## Authors

- Cian Dorr
- Ethan Russo

## License

MIT License - see LICENSE file for details.

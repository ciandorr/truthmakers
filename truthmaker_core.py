"""
truthmaker_core.py

Core data structures and operations for Kit Fine's truthmaker semantics.

This module implements bilateral propositions using a sophisticated encoding where:
- States are elements of a join-semilattice (configurable state space)
- Regular sets are encoded as (antichain, top_state) pairs
- Bilateral propositions are pairs of regular sets: (verifiers, falsifiers)

The state space can be:
- Auto-generated from atoms (default: nonempty subsets of {'r','g','b'})
- Auto-generated including empty state
- Custom: user-defined states with explicit partial order

Author: Refactored from truthmakers_fancy.py
"""

import sys
from functools import reduce, lru_cache
from itertools import product, combinations
import networkx as nx
from tqdm import tqdm


# ============================================================================
# SECTION 0: STATE SPACE (Join-Semilattice)
# ============================================================================

class StateSpace:
    """
    Represents a join-semilattice of states with precomputed operations.

    A state space consists of:
    - A set of state names (strings)
    - A partial order (≤) on the states
    - A binary join operation (fusion/supremum)

    The join operation and ordering are precomputed for O(1) lookup.
    """

    def __init__(self):
        """Create an empty state space (use factory methods to populate)."""
        self._states = set()           # State names
        self._order = {}               # state -> set of states it's ≤ to
        self._join = {}                # (a,b) -> join(a,b) where a <= b lexically
        self._has_empty = False
        self._is_default = True        # True = auto-generated from atoms
        self._atoms = ""               # Original atoms (for default mode)

    @classmethod
    def from_atoms(cls, atom_string, include_empty=False):
        """
        Generate power-set state space from atom characters.

        Args:
            atom_string: e.g., "rgb" for atoms r, g, b
            include_empty: Whether to include empty state (named "∅")

        Returns:
            StateSpace with all nonempty (or all) subsets as states
        """
        space = cls()
        space._atoms = atom_string
        atoms = set(atom_string)

        # Validate: "-" is reserved for the empty/bottom state name
        if '\u25A1' in atoms or '\u25A0' in atoms:
            raise ValueError(
                "The characters '\u25A1' and 'u25AO' cannot be used as an atom name because they are reserved for the bottom and top states. Please choose different atom names."
            )

        # Generate all subsets as frozensets
        all_subsets = []
        if include_empty:
            all_subsets.append(frozenset())
            space._has_empty = True

        for r in range(1, len(atoms) + 1):
            for combo in combinations(atoms, r):
                all_subsets.append(frozenset(combo))

        # Create canonical names for each subset
        # Sort atoms by their position in the input string (not alphabetically)
        def name_of(fs):
            if not fs:
                return "\u25A1"
            if fs == frozenset(atoms):
                return "\u25A0"
            return "".join(sorted(fs, key=lambda c: atom_string.index(c)))

        # Register all states
        for fs in all_subsets:
            space._states.add(name_of(fs))

        # Compute order (subset relation)
        for s1 in all_subsets:
            n1 = name_of(s1)
            space._order[n1] = set()
            for s2 in all_subsets:
                if s1 <= s2:  # subset relation
                    space._order[n1].add(name_of(s2))

        # Compute joins (union)
        for s1 in all_subsets:
            for s2 in all_subsets:
                n1, n2 = name_of(s1), name_of(s2)
                key = (n1, n2) if n1 <= n2 else (n2, n1)
                if key not in space._join:
                    space._join[key] = name_of(s1 | s2)

        space._is_default = True
        return space

    @classmethod
    def from_relations(cls, states, relations):
        """
        Build custom state space from explicit state names and ≤ relations.

        Args:
            states: List of state names (e.g., ["a", "b", "ab"])
            relations: List of (smaller, larger) pairs (e.g., [("a", "ab"), ("b", "ab")])

        Returns:
            StateSpace with auto-completed semilattice structure
        """
        space = cls()
        space._states = set(states)
        space._is_default = False

        # Initialize order with reflexive closure
        for s in states:
            space._order[s] = {s}

        # Add explicit relations
        for (a, b) in relations:
            if a not in space._states:
                raise ValueError(f"Unknown state in relation: {a}")
            if b not in space._states:
                raise ValueError(f"Unknown state in relation: {b}")
            space._order[a].add(b)

        # Compute transitive closure
        space._compute_transitive_closure()

        # Auto-complete to semilattice (add missing joins)
        space._complete_semilattice()

        return space

    def _compute_transitive_closure(self):
        """Floyd-Warshall transitive closure for the partial order."""
        states_list = list(self._states)
        n = len(states_list)
        idx = {s: i for i, s in enumerate(states_list)}

        # Build reachability matrix
        reach = [[False] * n for _ in range(n)]
        for a in states_list:
            for b in self._order.get(a, set()):
                reach[idx[a]][idx[b]] = True

        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if reach[i][k] and reach[k][j]:
                        reach[i][j] = True

        # Update order from reachability
        for i, a in enumerate(states_list):
            self._order[a] = set()
            for j, b in enumerate(states_list):
                if reach[i][j]:
                    self._order[a].add(b)

    def _complete_semilattice(self):
        """
        Auto-complete the partial order to a valid join-semilattice.

        For each pair without a join, create a synthetic state and add it.
        """
        max_iterations = 100
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            added_any = False

            states_list = list(self._states)
            for i, a in enumerate(states_list):
                for b in states_list[i:]:
                    # Check if join exists
                    key = (a, b) if a <= b else (b, a)
                    if key in self._join:
                        continue

                    # Find upper bounds
                    upper_bounds = []
                    for c in self._states:
                        if self.leq(a, c) and self.leq(b, c):
                            upper_bounds.append(c)

                    if not upper_bounds:
                        # No upper bound - create synthetic join
                        new_name = f"({a}⊔{b})"
                        self._states.add(new_name)
                        self._order[new_name] = {new_name}

                        # New state is above both a and b
                        self._order[a].add(new_name)
                        self._order[b].add(new_name)

                        # Propagate: anything above a and b is above new_name
                        for c in self._states:
                            if c != new_name and self.leq(a, c) and self.leq(b, c):
                                self._order[new_name].add(c)

                        self._join[key] = new_name
                        added_any = True
                    else:
                        # Find minimum upper bound (join)
                        min_ub = None
                        for ub in upper_bounds:
                            is_min = True
                            for other in upper_bounds:
                                if other != ub and self.leq(other, ub) and not self.leq(ub, other):
                                    is_min = False
                                    break
                            if is_min:
                                if min_ub is None:
                                    min_ub = ub
                                # Multiple minimal upper bounds means we need their join
                                # (will be handled in next iteration)

                        if min_ub:
                            self._join[key] = min_ub

            if not added_any:
                break

        # Recompute transitive closure after additions
        if iteration > 1:
            self._compute_transitive_closure()

    def join(self, a, b):
        """
        Return join (supremum/fusion) of two states.

        Args:
            a: State name
            b: State name

        Returns:
            State name of join(a, b)
        """
        key = (a, b) if a <= b else (b, a)
        if key not in self._join:
            raise ValueError(f"No join defined for ({a}, {b})")
        return self._join[key]

    def leq(self, a, b):
        """Return True if a ≤ b (a is part of b)."""
        return b in self._order.get(a, set())

    def lt(self, a, b):
        """Return True if a < b (proper part)."""
        return a != b and self.leq(a, b)

    def all_states(self):
        """Return set of all state names."""
        return self._states.copy()

    def state_size(self, state_name):
        """
        Return a notion of 'size' for sorting.

        For default mode: number of atoms
        For custom mode: number of states below this one
        """
        if self._is_default:
            return len(state_name) if state_name != "∅" else 0
        else:
            return sum(1 for s in self._states if self.lt(s, state_name))

    def hasse_edges(self):
        """
        Return edges for Hasse diagram (covering relations).

        Returns list of (lower, upper) pairs where lower < upper and
        there's no intermediate state.
        """
        edges = []
        for a in self._states:
            for b in self._order.get(a, set()):
                if a == b:
                    continue
                # Check if there's an intermediate state
                is_covering = True
                for c in self._states:
                    if c != a and c != b:
                        if self.leq(a, c) and self.leq(c, b):
                            is_covering = False
                            break
                if is_covering:
                    edges.append((a, b))
        return edges

    def validate(self):
        """
        Validate the state space structure.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check reflexivity
        for a in self._states:
            if not self.leq(a, a):
                errors.append(f"Reflexivity violated: {a} ≰ {a}")

        # Check antisymmetry
        for a in self._states:
            for b in self._states:
                if a != b and self.leq(a, b) and self.leq(b, a):
                    errors.append(f"Antisymmetry violated: {a} ≤ {b} and {b} ≤ {a}")

        # Check that joins exist
        for a in self._states:
            for b in self._states:
                key = (a, b) if a <= b else (b, a)
                if key not in self._join:
                    errors.append(f"Missing join: {a} ⊔ {b}")

        return errors


# Global state space (module-level)
_current_state_space = None


def get_state_space():
    """Get the current state space, creating default if needed."""
    global _current_state_space
    if _current_state_space is None:
        # Default: rgb atoms, no empty state
        _current_state_space = StateSpace.from_atoms("rgb", include_empty=False)
    return _current_state_space


def set_state_space(space):
    """Set the global state space."""
    global _current_state_space
    _current_state_space = space
    # Clear caches that depend on state space
    minimize_cached.cache_clear()


def reset_state_space():
    """Reset to default (will be re-created on next access)."""
    global _current_state_space
    _current_state_space = None
    minimize_cached.cache_clear()


# ============================================================================
# SECTION 1: BASIC DATA STRUCTURES
# ============================================================================

class State:
    """
    Represents a state in the current state space.

    States are identified by name (string) and delegate operations to
    the global StateSpace for fusion (join) and parthood (ordering).

    For backward compatibility, can also be created from frozensets
    when using the default (atom-based) state space.
    """

    def __init__(self, name_or_atoms):
        """
        Create a state.

        Args:
            name_or_atoms: Either:
                - A state name (string) matching a state in the current space
                - A frozenset of atoms (for backward compatibility in default mode)
        """
        space = get_state_space()

        if isinstance(name_or_atoms, str):
            # Direct name - use as-is
            self.name = name_or_atoms
        elif isinstance(name_or_atoms, frozenset):
            # Backward compat: convert frozenset to canonical name
            if space._is_default:
                if not name_or_atoms and space._has_empty:
                    self.name = "\u25A1"  # Empty/bottom state symbol
                elif not name_or_atoms:
                    raise ValueError("Empty state not allowed in current state space")
                else:
                    # Sort by position in atom_string (input order), not alphabetically
                    atom_string = space._atoms
                    self.name = "".join(sorted(name_or_atoms, key=lambda c: atom_string.index(c)))
            else:
                raise ValueError("Custom state space requires state names, not frozensets")
        elif isinstance(name_or_atoms, (set, list)):
            # Also accept set/list for backward compat
            fs = frozenset(name_or_atoms)
            if space._is_default:
                if not fs and space._has_empty:
                    self.name = "\u25A1"  # Empty/bottom state symbol
                elif not fs:
                    raise ValueError("Empty state not allowed in current state space")
                else:
                    # Sort by position in atom_string (input order), not alphabetically
                    atom_string = space._atoms
                    self.name = "".join(sorted(fs, key=lambda c: atom_string.index(c)))
            else:
                raise ValueError("Custom state space requires state names, not sets")
        else:
            raise TypeError(f"Expected str, frozenset, set, or list, got {type(name_or_atoms)}")

        # Validate state exists
        if self.name not in space._states:
            raise ValueError(f"Unknown state: '{self.name}' (valid states: {sorted(space._states)})")

    @property
    def atoms(self):
        """
        Backward compatibility: return frozenset of characters in name.

        Only meaningful for default (atom-based) state spaces.
        """
        space = get_state_space()
        if space._is_default:
            if self.name == "∅":
                return frozenset()
            return frozenset(self.name)
        else:
            # For custom spaces, return frozenset of the name itself
            return frozenset([self.name])

    def __repr__(self):
        return f"State({self.name})"

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return self.name == other.name

    def __or__(self, other):
        """Fuse two states (join in the state space)"""
        space = get_state_space()
        return State(space.join(self.name, other.name))

    def __le__(self, other):
        """Parthood relation (≤ in the state space)"""
        space = get_state_space()
        return space.leq(self.name, other.name)

    def __lt__(self, other):
        """Proper parthood relation (< in the state space)"""
        space = get_state_space()
        return space.lt(self.name, other.name)

    def to_string(self):
        """Convert state to string"""
        return self.name

    @classmethod
    def from_string(cls, string):
        """
        Create state from string.

        For default (atom-based) state spaces: string is interpreted as
        atom characters and converted to canonical name.
        For custom state spaces: string is the state name directly.
        """
        string = string.strip()
        space = get_state_space()

        if space._is_default:
            # Canonical name uses input order (from atom_string), not alphabetical
            if string == "∅" or string == "" or string == "-" or string == "\u25A1":
                if space._has_empty:
                    return cls("\u25A1")
                else:
                    raise ValueError("Empty state not allowed in current state space")
            # Handle top state (full fusion of all atoms)
            if string == "\u25A0":
                return cls("\u25A0")
            atom_string = space._atoms
            canonical = "".join(sorted(string, key=lambda c: atom_string.index(c)))
            return cls(canonical)
        else:
            # Custom space: use name directly
            return cls(string)


class RegularSet:
    """
    Encodes a regular set of states as an (antichain, top_state) pair, or null (empty set).

    A non-null regular set X is represented by:
    - antichain: The minimal elements of X (no element is a proper part of another)
    - top_state: The maximal element of X (fusion of all elements)

    A null RegularSet represents the empty set of states.
    This is useful for propositions that have no verifiers or no falsifiers.

    This encoding is more efficient than storing all states in X.
    """

    # Singleton for null instance
    _null_instance = None

    def __init__(self, antichain, top_state):
        """
        Create a regular set.

        Args:
            antichain: frozenset of States (minimal elements), or None for null
            top_state: State (maximal element), or None for null

        For null RegularSet, use RegularSet.null() factory method instead.
        """
        if antichain is None and top_state is None:
            # Null regular set (empty set of states)
            self._is_null = True
            self.antichain = None
            self.top_state = None
        else:
            self._is_null = False
            if isinstance(antichain, (list, set)):
                antichain = frozenset(antichain)
            self.antichain = antichain
            self.top_state = top_state

    @classmethod
    def null(cls):
        """Return the null (empty) regular set."""
        if cls._null_instance is None:
            cls._null_instance = cls(None, None)
        return cls._null_instance

    @property
    def is_null(self):
        """Return True if this is the null (empty) regular set."""
        return self._is_null

    def __repr__(self):
        return f"RegularSet({self.to_string()})"

    def __str__(self):
        return self.to_string()

    def __hash__(self):
        if self._is_null:
            return hash(None)
        return hash((self.antichain, self.top_state))

    def __eq__(self, other):
        if not isinstance(other, RegularSet):
            return False
        if self._is_null and other._is_null:
            return True
        if self._is_null or other._is_null:
            return False
        return self.antichain == other.antichain and self.top_state == other.top_state

    def to_string(self):
        """Convert to string: 'r,gb;rgb' (antichain;top) or '0' for null"""
        if self._is_null:
            return "0"
        antichain_str = stateset_to_string(self.antichain)
        top_str = self.top_state.to_string()
        return f"{antichain_str};{top_str}"

    @classmethod
    def from_string(cls, string):
        """
        Parse from string: 'r,gb;rgb' or 'r g;rgb' or '0' for null

        Args:
            string: Format is 'antichain_states;top_state' or '0' for null
                   where antichain_states can be comma or space separated
        """
        string = string.strip()

        # Check for null
        if string == '0':
            return cls.null()

        parts = string.split(';')
        if len(parts) != 2:
            raise ValueError(f"Invalid RegularSet string format: {string}")

        antichain_str, top_str = parts
        antichain = stateset_from_string(antichain_str)
        top_state = State.from_string(top_str)

        return cls(antichain, top_state)


class BilateralProposition:
    """
    A bilateral proposition: a pair of regular sets (verifiers, falsifiers).

    In Fine's truthmaker semantics:
    - Verifiers are states that make the proposition true
    - Falsifiers are states that make the proposition false
    """

    def __init__(self, verifiers, falsifiers):
        """
        Create a bilateral proposition.

        Args:
            verifiers: RegularSet
            falsifiers: RegularSet
        """
        self.verifiers = verifiers
        self.falsifiers = falsifiers

    def __repr__(self):
        return f"BilateralProposition({self.to_string()})"

    def __str__(self):
        return self.to_string()

    def __hash__(self):
        return hash((self.verifiers, self.falsifiers))

    def __eq__(self, other):
        if not isinstance(other, BilateralProposition):
            return False
        return self.verifiers == other.verifiers and self.falsifiers == other.falsifiers

    def to_string(self):
        """Convert to string: '(r,gb;rgb | b,rg;rgb)'"""
        return f"({self.verifiers.to_string()} | {self.falsifiers.to_string()})"

    @classmethod
    def from_string(cls, string):
        """
        Parse from string: 'r;r|gb;gb' or '(r;r | gb;gb)'

        Format: verifiers|falsifiers where each side is a RegularSet string
        """
        # Remove parentheses if present
        string = string.strip()
        if string.startswith('(') and string.endswith(')'):
            string = string[1:-1]

        # Split by | to get verifiers and falsifiers
        parts = string.split('|')
        if len(parts) != 2:
            raise ValueError(f"Invalid BilateralProposition format: {string}")

        verifiers = RegularSet.from_string(parts[0].strip())
        falsifiers = RegularSet.from_string(parts[1].strip())

        return cls(verifiers, falsifiers)

    def is_definite(self):
        """Check if this is a definite proposition (verifier set is singleton)"""
        # A definite is when top_state is in the antichain (only possible if singleton)
        # Null verifiers cannot be definite
        if self.verifiers.is_null:
            return False
        return self.verifiers.top_state in self.verifiers.antichain

    def is_diamond(self):
        """Check if this is a diamond proposition (antichain has exactly one element)"""
        # Null verifiers cannot be diamond
        if self.verifiers.is_null:
            return False
        return len(self.verifiers.antichain) == 1


# ============================================================================
# SECTION 2: HELPER FUNCTIONS FOR STATES AND STATE SETS
# ============================================================================

def fuse(pair):
    """
    Fuse two states (union).

    Args:
        pair: tuple of (State, State)

    Returns:
        State: fusion of the two states
    """
    return pair[0] | pair[1]


def isproperpart(first, second):
    """Check if first is a proper part of second"""
    return first < second


@lru_cache(maxsize=10000)
def minimize_cached(stateset_tuple):
    """
    Cached version of minimize for performance.
    Takes a tuple of states (for hashability).
    """
    stateset = set(stateset_tuple)
    # Sort by state size (larger first) - uses StateSpace for custom spaces
    space = get_state_space()
    listset = sorted(stateset, key=lambda s: space.state_size(s.name), reverse=True)
    n = 0
    while n < len(listset):
        for test in listset:
            if isproperpart(test, listset[n]):
                listset.pop(n)
                n = n - 1
                break
        n = n + 1
    return frozenset(listset)


def minimize(stateset):
    """
    Extract minimal elements from a set of states (antichain).

    Returns the subset containing just the minimal elements - those that
    have no proper part in the set.

    Args:
        stateset: iterable of States

    Returns:
        frozenset of States (the antichain)
    """
    if isinstance(stateset, frozenset):
        stateset_tuple = tuple(sorted(stateset, key=lambda s: s.to_string()))
    else:
        stateset_tuple = tuple(sorted(stateset, key=lambda s: s.to_string()))

    return minimize_cached(stateset_tuple)


def stateset_to_string(stateset):
    """Convert a set of states to comma-separated string: 'r,gb,rgb'"""
    space = get_state_space()
    def sort_key(state):
        # Sort by size descending, then by name
        return (-space.state_size(state.name), state.to_string())

    sorted_states = sorted(stateset, key=sort_key)
    return ",".join([state.to_string() for state in sorted_states])


def stateset_from_string(string):
    """
    Parse a set of states from string.

    Accepts either comma-separated ('r,gb,rgb') or space-separated ('r gb rgb')
    """
    string = string.strip()
    if not string:
        return frozenset()

    # Try comma-separated first, fall back to space-separated
    if ',' in string:
        state_strs = string.split(',')
    else:
        state_strs = string.split()

    states = [State.from_string(s.strip()) for s in state_strs if s.strip()]
    return minimize(states)


# ============================================================================
# SECTION 3: OPERATIONS ON REGULAR SETS
# ============================================================================

def bottomwedge(first_antichain, second_antichain):
    """
    Compute the antichain of the conjunction of two regular sets.

    The conjunction of two regular sets X and Y is formed by taking all
    fusions of elements from X with elements from Y, then minimizing.
    """
    fusions = map(fuse, product(first_antichain, second_antichain))
    return minimize(fusions)


def bottomvee(first_antichain, second_antichain):
    """
    Compute the antichain of the disjunction of two regular sets.

    The disjunction is just the union of the two antichains, minimized.
    """
    return minimize(first_antichain | second_antichain)


def wedge(first, second):
    """
    Conjunction of two regular sets (as encoded pairs).

    Args:
        first: RegularSet
        second: RegularSet

    Returns:
        RegularSet: conjunction of the two

    Null handling: null ∧ X = null, X ∧ null = null
    """
    # Null handling: conjunction with null is null
    if first.is_null or second.is_null:
        return RegularSet.null()

    new_antichain = bottomwedge(first.antichain, second.antichain)
    new_top = first.top_state | second.top_state
    return RegularSet(new_antichain, new_top)


def vee(first, second):
    """
    Disjunction of two regular sets (as encoded pairs).

    Args:
        first: RegularSet
        second: RegularSet

    Returns:
        RegularSet: disjunction of the two

    Null handling: null ∨ X = X, X ∨ null = X, null ∨ null = null
    """
    # Null handling: disjunction with null returns the other
    if first.is_null and second.is_null:
        return RegularSet.null()
    if first.is_null:
        return second
    if second.is_null:
        return first

    new_antichain = bottomvee(first.antichain, second.antichain)
    new_top = first.top_state | second.top_state
    return RegularSet(new_antichain, new_top)


def singleton_regular_set(state):
    """
    Create a singleton RegularSet containing just one state.

    Args:
        state: State object

    Returns:
        RegularSet with antichain={state} and top_state=state
    """
    return RegularSet(frozenset([state]), state)


def state_in_regular_set(state, regular_set):
    """
    Check if a state is contained in a regular set.

    A state S is in the regular set if:
    - There exists an antichain element A such that A ≤ S
    - AND S ≤ top_state

    In other words, S is "between" some antichain element and the top.

    Args:
        state: State object to check
        regular_set: RegularSet to check membership in

    Returns:
        bool: True if state is in the regular set
    """
    if regular_set.is_null:
        return False

    # Check if state is below or equal to top
    if not (state <= regular_set.top_state):
        return False

    # Check if some antichain element is below or equal to state
    for antichain_elem in regular_set.antichain:
        if antichain_elem <= state:
            return True

    return False


def all_states_in_regular_set(regular_set):
    """
    Return all states contained in a regular set.

    Args:
        regular_set: RegularSet

    Returns:
        set of State objects contained in the regular set
    """
    if regular_set.is_null:
        return set()

    space = get_state_space()
    result = set()

    for state_name in space.all_states():
        state = State(state_name)
        if state_in_regular_set(state, regular_set):
            result.add(state)

    return result


# ============================================================================
# SECTION 4: OPERATIONS ON BILATERAL PROPOSITIONS
# ============================================================================

def conjoin(first, second):
    """
    Conjunction of two bilateral propositions.

    Wedges the verifier sets and vees the falsifier sets.

    Args:
        first: BilateralProposition
        second: BilateralProposition

    Returns:
        BilateralProposition: p ∧ q
    """
    new_verifiers = wedge(first.verifiers, second.verifiers)
    new_falsifiers = vee(first.falsifiers, second.falsifiers)
    return BilateralProposition(new_verifiers, new_falsifiers)


def disjoin(first, second):
    """
    Disjunction of two bilateral propositions.

    Vees the verifier sets and wedges the falsifier sets.

    Args:
        first: BilateralProposition
        second: BilateralProposition

    Returns:
        BilateralProposition: p ∨ q
    """
    new_verifiers = vee(first.verifiers, second.verifiers)
    new_falsifiers = wedge(first.falsifiers, second.falsifiers)
    return BilateralProposition(new_verifiers, new_falsifiers)


def negate(prop):
    """
    Negation of a bilateral proposition.

    Simply swaps verifiers and falsifiers.

    Args:
        prop: BilateralProposition

    Returns:
        BilateralProposition: ¬p
    """
    return BilateralProposition(prop.falsifiers, prop.verifiers)


# ============================================================================
# SECTION 5: PROPOSITION SET WITH CLOSURE OPERATIONS
# ============================================================================

class PropositionSet:
    """
    A collection of bilateral propositions with closure operations.

    Provides methods to generate closures under negation, conjunction,
    and disjunction operations.
    """

    def __init__(self, propositions=None):
        """
        Create a proposition set.

        Args:
            propositions: list of BilateralProposition objects
        """
        if propositions is None:
            propositions = []

        # Use both set and list for O(1) membership testing + indexing
        self._prop_set = set(propositions)
        self._prop_list = list(propositions)

    def __len__(self):
        return len(self._prop_list)

    def __iter__(self):
        return iter(self._prop_list)

    def __getitem__(self, index):
        return self._prop_list[index]

    def add(self, prop):
        """Add a proposition if not already present"""
        if prop not in self._prop_set:
            self._prop_set.add(prop)
            self._prop_list.append(prop)
            return True
        return False

    def to_list(self):
        """Return as a list"""
        return list(self._prop_list)

    def negclose(self, verbose=False):
        """
        Close under negation.

        For each proposition p, adds ¬p if not already present.

        Args:
            verbose: If True, print progress to stderr

        Returns:
            self (for chaining)
        """
        n = 0
        while n < len(self._prop_list):
            new = negate(self._prop_list[n])
            if self.add(new):
                if verbose:
                    print(f"{len(self)}: {new.to_string()} = ~{n}",
                          file=sys.stderr, end="\r")
            n += 1

        if verbose:
            print(file=sys.stderr)  # New line

        return self

    def conjclose(self, verbose=False):
        """
        Close under conjunction.

        For each pair of propositions p, q, adds p ∧ q if not already present.

        Args:
            verbose: If True, print progress to stderr

        Returns:
            self (for chaining)
        """
        n = 0
        while n < len(self._prop_list):
            m = 0
            while m < n:
                new = conjoin(self._prop_list[m], self._prop_list[n])
                if self.add(new):
                    if verbose:
                        print(f"{len(self)}: {new.to_string()} = {n} ^ {m}",
                              file=sys.stderr, end="\r")
                m += 1
            n += 1

        if verbose:
            print(file=sys.stderr)  # New line

        return self

    def disjclose(self, verbose=False):
        """
        Close under disjunction.

        For each pair of propositions p, q, adds p ∨ q if not already present.

        Args:
            verbose: If True, print progress to stderr

        Returns:
            self (for chaining)
        """
        n = 0
        while n < len(self._prop_list):
            m = 0
            while m < n:
                new = disjoin(self._prop_list[m], self._prop_list[n])
                if self.add(new):
                    if verbose:
                        print(f"{len(self)}: {new.to_string()} = {n} v {m}",
                              file=sys.stderr, end="\r")
                m += 1
            n += 1

        if verbose:
            print(file=sys.stderr)  # New line

        return self

    def close(self, verbose=False):
        """
        Close under all operations: negation, conjunction, then disjunction.

        Due to distributive and De Morgan laws for regular bilateral propositions,
        we can apply operations in this order to get the full closure.

        Args:
            verbose: If True, print progress to stderr

        Returns:
            self (for chaining)
        """
        if verbose:
            print("Closing under negation...", file=sys.stderr)
        self.negclose(verbose=verbose)

        if verbose:
            print(f"After negation: {len(self)} propositions", file=sys.stderr)
            print("Closing under conjunction...", file=sys.stderr)
        self.conjclose(verbose=verbose)

        if verbose:
            print(f"After conjunction: {len(self)} propositions", file=sys.stderr)
            print("Closing under disjunction...", file=sys.stderr)
        self.disjclose(verbose=verbose)

        if verbose:
            print(f"Final: {len(self)} propositions", file=sys.stderr)

        return self


# ============================================================================
# SECTION 6: ANALYSIS UTILITIES
# ============================================================================

def definites(prop_set):
    """
    Filter to definite propositions.

    A definite proposition has a singleton verifier set.

    Args:
        prop_set: PropositionSet or list of BilateralPropositions

    Returns:
        list of BilateralPropositions
    """
    return [p for p in prop_set if p.is_definite()]


def diamonds(prop_set):
    """
    Filter to diamond propositions.

    A diamond proposition has an antichain of length 1.

    Args:
        prop_set: PropositionSet or list of BilateralPropositions

    Returns:
        list of BilateralPropositions
    """
    return [p for p in prop_set if p.is_diamond()]


def L_reorganize(prop_set):
    """
    Group propositions by L-equivalence class.

    L-equivalence class is defined by the pair of antichains (verifier, falsifier).
    Returns a list of [L_class, [list of M_classes]] pairs.

    Args:
        prop_set: PropositionSet or list of BilateralPropositions

    Returns:
        list of [L_class, M_classes] where:
            L_class = (verifier_antichain, falsifier_antichain)
            M_classes = list of (verifier_top, falsifier_top) tuples
    """
    result = []
    for prop in prop_set:
        # Handle null RegularSets: use empty frozenset for null antichains
        v_antichain = prop.verifiers.antichain if not prop.verifiers.is_null else frozenset()
        f_antichain = prop.falsifiers.antichain if not prop.falsifiers.is_null else frozenset()
        L_class = (v_antichain, f_antichain)
        M_class = (prop.verifiers.top_state, prop.falsifiers.top_state)

        # Find if this L_class already exists
        found = False
        for entry in result:
            if entry[0] == L_class:
                entry[1].append(M_class)
                found = True
                break

        if not found:
            result.append([L_class, [M_class]])

    return result


def M_reorganize(prop_set):
    """
    Group propositions by M-equivalence class.

    M-equivalence class is defined by the pair of top states (verifier, falsifier).
    Returns a list of [M_class, [list of L_classes]] pairs.

    Args:
        prop_set: PropositionSet or list of BilateralPropositions

    Returns:
        list of [M_class, L_classes] where:
            M_class = (verifier_top, falsifier_top)
            L_classes = list of (verifier_antichain, falsifier_antichain) tuples
    """
    result = []
    for prop in prop_set:
        # Handle null RegularSets: use empty frozenset for null antichains
        v_antichain = prop.verifiers.antichain if not prop.verifiers.is_null else frozenset()
        f_antichain = prop.falsifiers.antichain if not prop.falsifiers.is_null else frozenset()
        L_class = (v_antichain, f_antichain)
        # M_class keeps None for null top_states (Option[State] pattern)
        M_class = (prop.verifiers.top_state, prop.falsifiers.top_state)

        # Find if this M_class already exists
        found = False
        for entry in result:
            if entry[0] == M_class:
                entry[1].append(L_class)
                found = True
                break

        if not found:
            result.append([M_class, [L_class]])

    return result


def top(prop_set):
    """
    Compute the disjunction of all propositions in the set.

    Args:
        prop_set: PropositionSet or list of BilateralPropositions

    Returns:
        BilateralProposition
    """
    props = list(prop_set)
    if not props:
        raise ValueError("Cannot compute top of empty proposition set")
    return reduce(disjoin, props)


def bottom(prop_set):
    """
    Compute the conjunction of all propositions in the set.

    Args:
        prop_set: PropositionSet or list of BilateralPropositions

    Returns:
        BilateralProposition
    """
    props = list(prop_set)
    if not props:
        raise ValueError("Cannot compute bottom of empty proposition set")
    return reduce(conjoin, props)


# ============================================================================
# SECTION 7: POSET CONSTRUCTION
# ============================================================================

class TruthmakerPoset:
    """
    A partially ordered set (poset) of propositions.

    Wraps a NetworkX DiGraph with truthmaker-specific methods.
    The ordering can be either conjunction-based (≤∧) or disjunction-based (≤∨).
    """

    def __init__(self, prop_set, order_type='conjunction'):
        """
        Build a poset from a set of propositions.

        Args:
            prop_set: PropositionSet or list of BilateralPropositions
            order_type: 'conjunction' or 'disjunction'
                - 'conjunction': p ≤ q iff p ∧ q = q
                - 'disjunction': p ≤ q iff p ∨ q = q
        """
        self.prop_list = list(prop_set)
        self.order_type = order_type

        # Build the poset
        if order_type == 'conjunction':
            test_fn = lambda p, q: conjoin(p, q) == q
        elif order_type == 'disjunction':
            test_fn = lambda p, q: disjoin(p, q) == q
        else:
            raise ValueError(f"Unknown order_type: {order_type}")

        # Build directed graph
        self.graph = nx.DiGraph()

        # Add nodes (use index as node ID for simplicity)
        for i in range(len(self.prop_list)):
            self.graph.add_node(i)

        # Add edges
        print(f"Building {order_type} poset...", file=sys.stderr)
        for i in tqdm(range(len(self.prop_list))):
            for j in range(len(self.prop_list)):
                if i != j and test_fn(self.prop_list[i], self.prop_list[j]):
                    self.graph.add_edge(i, j)

        print(f"Poset built: {len(self.prop_list)} nodes, {self.graph.number_of_edges()} edges",
              file=sys.stderr)

    def transitive_reduction(self):
        """
        Return the transitive reduction of this poset.

        The transitive reduction removes redundant edges while preserving
        the reachability relation. This is what you want for Hasse diagrams.

        Returns:
            TruthmakerPoset with reduced graph
        """
        reduced = TruthmakerPoset.__new__(TruthmakerPoset)
        reduced.prop_list = self.prop_list
        reduced.order_type = self.order_type

        print("Computing transitive reduction...", file=sys.stderr)
        reduced.graph = nx.transitive_reduction(self.graph)
        print(f"Reduced to {reduced.graph.number_of_edges()} edges", file=sys.stderr)

        return reduced

    def get_proposition(self, node_id):
        """Get the proposition associated with a node ID"""
        return self.prop_list[node_id]

    def predecessors(self, node_id):
        """Get predecessor node IDs (nodes with edges pointing to this node)"""
        return list(self.graph.predecessors(node_id))

    def successors(self, node_id):
        """Get successor node IDs (nodes this node points to)"""
        return list(self.graph.successors(node_id))


# ============================================================================
# SECTION 8: CONVENIENCE FUNCTIONS
# ============================================================================

def parse_propositions(strings):
    """
    Parse multiple propositions from strings.

    Args:
        strings: list of proposition strings

    Returns:
        PropositionSet
    """
    props = [BilateralProposition.from_string(s) for s in strings]
    return PropositionSet(props)


# ============================================================================
# SECTION 9: EQUIVALENCE CLASSES
# ============================================================================

def get_L_class(prop):
    """
    Get the L-equivalence class of a proposition.

    L-class is determined by the antichains: (verifier_antichain, falsifier_antichain)

    Args:
        prop: BilateralProposition

    Returns:
        tuple of (frozenset of verifier antichain states, frozenset of falsifier antichain states)
        For null RegularSets, the antichain is treated as empty frozenset.
    """
    # For null RegularSets, treat antichain as empty frozenset
    v_antichain = prop.verifiers.antichain if not prop.verifiers.is_null else frozenset()
    f_antichain = prop.falsifiers.antichain if not prop.falsifiers.is_null else frozenset()
    return (v_antichain, f_antichain)


def get_M_class(prop):
    """
    Get the M-equivalence class of a proposition.

    M-class is determined by the tops: (verifier_top, falsifier_top)

    Args:
        prop: BilateralProposition

    Returns:
        tuple of (verifier_top State, falsifier_top State)
    """
    return (prop.verifiers.top_state, prop.falsifiers.top_state)


def group_by_L_class(prop_set):
    """
    Group propositions by L-equivalence class.

    Args:
        prop_set: PropositionSet or list of BilateralPropositions

    Returns:
        dict mapping L-class -> list of propositions
    """
    props = prop_set.to_list() if isinstance(prop_set, PropositionSet) else list(prop_set)

    result = {}
    for prop in props:
        L_class = get_L_class(prop)
        if L_class not in result:
            result[L_class] = []
        result[L_class].append(prop)

    return result


def group_by_M_class(prop_set):
    """
    Group propositions by M-equivalence class.

    Args:
        prop_set: PropositionSet or list of BilateralPropositions

    Returns:
        dict mapping M-class -> list of propositions
    """
    props = prop_set.to_list() if isinstance(prop_set, PropositionSet) else list(prop_set)

    result = {}
    for prop in props:
        M_class = get_M_class(prop)
        if M_class not in result:
            result[M_class] = []
        result[M_class].append(prop)

    return result


def L_class_to_string(L_class):
    """Convert L-class to readable string"""
    v_antichain, f_antichain = L_class
    # Handle empty frozensets (from null RegularSets) - display as "0"
    v_str = stateset_to_string(v_antichain) if v_antichain else "0"
    f_str = stateset_to_string(f_antichain) if f_antichain else "0"
    return f"L[{v_str}|{f_str}]"


def M_class_to_string(M_class):
    """Convert M-class to readable string"""
    v_top, f_top = M_class
    # Handle None top_state (from null RegularSets) - display as "0"
    v_str = v_top.to_string() if v_top is not None else "0"
    f_str = f_top.to_string() if f_top is not None else "0"
    return f"M[{v_str}|{f_str}]"


def build_L_lattice(prop_set):
    """
    Build the lattice structure of L-equivalence classes.

    The ordering is: L[a,b] ≤ L[c,d] iff:
      - bottomwedge(a, c) == a  (verifier antichains)
      - bottomvee(b, d) == b    (falsifier antichains)

    Args:
        prop_set: PropositionSet or list of BilateralPropositions

    Returns:
        NetworkX DiGraph where nodes are L-classes and edges represent the ordering
    """
    L_classes = list(group_by_L_class(prop_set).keys())

    G = nx.DiGraph()

    # Add nodes
    for L_class in L_classes:
        G.add_node(L_class)

    # Add edges for ordering relation
    for i, L1 in enumerate(L_classes):
        for L2 in L_classes[i+1:]:
            v1_antichain, f1_antichain = L1
            v2_antichain, f2_antichain = L2

            # Check if L1 ≤ L2
            if (bottomwedge(v1_antichain, v2_antichain) == v1_antichain and
                bottomvee(f1_antichain, f2_antichain) == f1_antichain):
                G.add_edge(L1, L2)
            # Check if L2 ≤ L1
            elif (bottomwedge(v2_antichain, v1_antichain) == v2_antichain and
                  bottomvee(f2_antichain, f1_antichain) == f2_antichain):
                G.add_edge(L2, L1)

    return G


def _option_state_leq_bottom(a, b):
    """
    Compare Option[State] values with None as bottom element (≤_1).

    - None ≤_1 anything
    - x ≤_1 None only if x is None

    Args:
        a: State or None
        b: State or None

    Returns:
        True if a ≤_1 b
    """
    if a is None:
        return True
    if b is None:
        return False
    return a <= b


def _option_state_leq_top(a, b):
    """
    Compare Option[State] values with None as top element (≤_2).

    - anything ≤_2 None
    - None ≤_2 x only if x is None

    Args:
        a: State or None
        b: State or None

    Returns:
        True if a ≤_2 b
    """
    if b is None:
        return True
    if a is None:
        return False
    return a <= b


def has_null_m_class(prop_set):
    """
    Check if any proposition in the set has a null verifier or falsifier.

    Args:
        prop_set: PropositionSet or list of BilateralPropositions

    Returns:
        True if any proposition has null verifiers or null falsifiers
    """
    for prop in prop_set:
        if prop.verifiers.is_null or prop.falsifiers.is_null:
            return True
    return False


def build_M_semilattice(prop_set, ordering_option=None):
    """
    Build the semilattice structure of M-equivalence classes.

    The ordering is: M[s,t] ≤ M[u,v] iff s ≤ u and t ≤ v
    (where ≤ is the parthood relation on states, allowing equality)

    When nulls are present (propositions with null verifiers or falsifiers),
    there are two possible orderings based on how None is treated:
    - ≤_1: None at bottom (None ≤ everything)
    - ≤_2: None at top (everything ≤ None)

    The two options are:
    - Option 'A': M[s,t] ≤_A M[u,v] iff s ≤_1 u and t ≤_2 v
    - Option 'B': M[s,t] ≤_B M[u,v] iff s ≤_2 u and t ≤_1 v

    These agree when no nulls are present (both reduce to componentwise ≤).

    Args:
        prop_set: PropositionSet or list of BilateralPropositions
        ordering_option: 'A' or 'B' (only needed when nulls present).
            If None and nulls are present, defaults to 'A'.

    Returns:
        NetworkX DiGraph where nodes are M-classes and edges represent the ordering
    """
    M_classes = list(group_by_M_class(prop_set).keys())

    G = nx.DiGraph()

    # Add nodes
    for M_class in M_classes:
        G.add_node(M_class)

    # Check if any null top_states are present
    nulls_present = any(
        v_top is None or f_top is None
        for v_top, f_top in M_classes
    )

    # Default ordering option if nulls present
    if nulls_present and ordering_option is None:
        ordering_option = 'A'

    # Select comparison functions based on ordering option
    if ordering_option == 'A':
        # Option A: verifiers use ≤_1 (None at bottom), falsifiers use ≤_2 (None at top)
        v_leq = _option_state_leq_bottom
        f_leq = _option_state_leq_top
    else:
        # Option B: verifiers use ≤_2 (None at top), falsifiers use ≤_1 (None at bottom)
        v_leq = _option_state_leq_top
        f_leq = _option_state_leq_bottom

    # Add edges for ordering relation
    for i, M1 in enumerate(M_classes):
        for M2 in M_classes[i+1:]:
            if M1 == M2:
                continue

            v1_top, f1_top = M1
            v2_top, f2_top = M2

            # M1 ≤ M2 iff v1 ≤ v2 and f1 ≤ f2 (using selected comparison functions)
            if v_leq(v1_top, v2_top) and f_leq(f1_top, f2_top):
                G.add_edge(M1, M2)
            elif v_leq(v2_top, v1_top) and f_leq(f2_top, f1_top):
                G.add_edge(M2, M1)

    return G

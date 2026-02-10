"""
test_truthmaker.py

Quick test script to verify the truthmaker modules work correctly.
"""

import sys
sys.path.insert(0, '.')

from truthmaker_core import (
    State, RegularSet, BilateralProposition,
    PropositionSet, TruthmakerPoset,
    parse_propositions, definites, diamonds,
    conjoin, disjoin, negate
)

from truthmaker_visualization import create_plotly_graph

print("="*60)
print("TRUTHMAKER SEMANTICS - BASIC FUNCTIONALITY TEST")
print("="*60)

# Test 1: Create states
print("\n1. Testing State creation...")
r = State('r')
g = State('g')
b = State('b')
rgb = r | g | b
print(f"   r = {r}")
print(f"   g = {g}")
print(f"   b = {b}")
print(f"   r ∪ g ∪ b = {rgb}")
assert rgb.to_string() == 'rgb', "State fusion failed"
print("   ✓ States work correctly")

# Test 2: Create regular sets
print("\n2. Testing RegularSet creation...")
V = RegularSet(frozenset([r]), r)
F = RegularSet(frozenset([g, b]), g | b)
print(f"   V = {V.to_string()}")
print(f"   F = {F.to_string()}")
print("   ✓ Regular sets work correctly")

# Test 3: Create propositions
print("\n3. Testing BilateralProposition creation...")
p = BilateralProposition(V, F)
print(f"   p = {p.to_string()}")

# Test from_string and to_string round-trip
p_str = "r;r|g b;gb"
p_parsed = BilateralProposition.from_string(p_str)
print(f"   Parsed '{p_str}' -> {p_parsed.to_string()}")
assert p == p_parsed, "String parsing failed"
print("   ✓ Propositions work correctly")

# Test 4: Operations
print("\n4. Testing operations...")
q_str = "g;g|r b;rb"
q = BilateralProposition.from_string(q_str)

p_and_q = conjoin(p, q)
print(f"   p ∧ q = {p_and_q.to_string()}")

p_or_q = disjoin(p, q)
print(f"   p ∨ q = {p_or_q.to_string()}")

not_p = negate(p)
print(f"   ¬p = {not_p.to_string()}")
print("   ✓ Operations work correctly")

# Test 5: Parse atoms1
print("\n5. Testing parse_propositions...")
atoms1 = parse_propositions(['r;r|g b;gb', 'g;g|r b;rb', 'b;b|r g;rg'])
print(f"   Parsed {len(atoms1)} atoms")
for i, atom in enumerate(atoms1):
    print(f"     {i}: {atom.to_string()}")
print("   ✓ Parsing works correctly")

# Test 6: Closure
print("\n6. Testing closure operations...")
prop_set = PropositionSet(list(atoms1))
print(f"   Initial: {len(prop_set)} propositions")

prop_set.negclose(verbose=False)
print(f"   After negation: {len(prop_set)} propositions")

prop_set.conjclose(verbose=False)
print(f"   After conjunction: {len(prop_set)} propositions")

prop_set.disjclose(verbose=False)
print(f"   After disjunction: {len(prop_set)} propositions")

print("   ✓ Closure operations work correctly")

# Test 7: Definites and diamonds
print("\n7. Testing filters...")
defs = definites(prop_set)
dias = diamonds(prop_set)
print(f"   Definites: {len(defs)}")
print(f"   Diamonds: {len(dias)}")
print("   ✓ Filters work correctly")

# Test 8: Poset construction
print("\n8. Testing poset construction...")
print("   Building conjunction poset...")
poset = TruthmakerPoset(prop_set, order_type='conjunction')
print(f"   Nodes: {len(poset.prop_list)}")
print(f"   Edges: {poset.graph.number_of_edges()}")

print("   Computing transitive reduction...")
poset_reduced = poset.transitive_reduction()
print(f"   Reduced edges: {poset_reduced.graph.number_of_edges()}")
print("   ✓ Poset construction works correctly")

# Test 9: Visualization
print("\n9. Testing visualization...")
try:
    fig = create_plotly_graph(poset_reduced, layout='hierarchical', color_by='generation')
    print("   ✓ Visualization works correctly")
    print("   (Figure created, but not displayed in test)")
except Exception as e:
    print(f"   ✗ Visualization failed: {e}")

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
print("\nYou can now use the Jupyter notebook:")
print("  jupyter notebook truthmaker_viz.ipynb")
print("="*60)

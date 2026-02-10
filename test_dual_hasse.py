"""
test_dual_hasse.py

Test script for the dual Hasse diagram visualization.
"""

import sys
sys.path.insert(0, '.')

from truthmaker_core import (
    parse_propositions, PropositionSet,
    build_L_lattice, build_M_semilattice,
    group_by_L_class, group_by_M_class
)

from truthmaker_visualization import create_dual_hasse_diagram

print("=" * 60)
print("DUAL HASSE DIAGRAM TEST")
print("=" * 60)

# Parse atoms1
print("\n1. Parsing atoms1...")
atoms1 = parse_propositions(['r;r|g b;gb', 'g;g|r b;rb', 'b;b|r g;rg'])
print(f"   Parsed {len(atoms1)} atoms")

# Generate closure
print("\n2. Generating closure...")
atoms1.close(verbose=False)
print(f"   Generated {len(atoms1)} propositions")

# Group by equivalence classes
print("\n3. Computing equivalence classes...")
L_groups = group_by_L_class(atoms1)
M_groups = group_by_M_class(atoms1)
print(f"   L-classes: {len(L_groups)}")
print(f"   M-classes: {len(M_groups)}")

# Build lattice structures
print("\n4. Building lattice structures...")
L_lattice = build_L_lattice(atoms1)
M_semilattice = build_M_semilattice(atoms1)
print(f"   L-lattice: {L_lattice.number_of_nodes()} nodes, {L_lattice.number_of_edges()} edges")
print(f"   M-semilattice: {M_semilattice.number_of_nodes()} nodes, {M_semilattice.number_of_edges()} edges")

# Create visualization
print("\n5. Creating dual Hasse diagram...")
try:
    fig = create_dual_hasse_diagram(atoms1, title="Dual Hasse Diagram - atoms1")
    print("   ✓ Visualization created successfully")

    # Save to HTML
    output_file = "dual_hasse_atoms1.html"
    fig.write_html(output_file)
    print(f"   ✓ Saved to {output_file}")
    print(f"\n   Open {output_file} in your browser to view!")

except Exception as e:
    print(f"   ✗ Visualization failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)

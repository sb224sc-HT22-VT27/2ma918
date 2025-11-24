"""
Exercise 1: Minimum Cost Flow Problems
Solve the logistics problem for a company in the Kronoberg region
"""

import numpy as np
from scipy.optimize import linprog


def solve_exercise_1_1():
    """
    Part 1: Formulate and solve the minimum cost network flow problem
    """

    # Define the network structure
    # Nodes: Älmhult(0), Markaryd(1), Liatorp(2), Osby(3), Ljungby(4), Alvesta(5), Växjö(6)

    # Edges and their properties: (from, to, cost, capacity)
    edges = [
        (0, 2, 18, 1000),  # Älmhult -> Liatorp
        (0, 3, 24, 1000),  # Älmhult -> Osby
        (0, 6, 62, 1000),  # Älmhult -> Växjö
        (3, 1, 29, 1000),  # Osby -> Markaryd
        (2, 4, 31, 500),  # Liatorp -> Ljungby
        (2, 6, 46, 1000),  # Liatorp -> Växjö
        (1, 4, 51, 2000),  # Markaryd -> Ljungby
        (4, 5, 42, 500),  # Ljungby -> Alvesta
        (5, 6, 20, 2000),  # Alvesta -> Växjö
    ]

    num_edges = len(edges)

    # Cost vector (coefficients to minimize)
    c = np.array([edge[2] for edge in edges])

    # Bounds for each edge (0 <= flow <= capacity)
    bounds = [(0, edge[3]) for edge in edges]

    # Supply and demand constraints (flow conservation at each node)
    # Supply: Älmhult=2500, Markaryd=1000
    # Demand: Ljungby=1000, Alvesta=500, Växjö=2000
    # Transshipment: Liatorp=0, Osby=0

    # A_eq matrix for flow conservation (rows = nodes, cols = edges)
    # Node order: Älmhult(0), Markaryd(1), Liatorp(2), Osby(3), Ljungby(4), Alvesta(5), Växjö(6)
    num_nodes = 7
    A_eq = np.zeros((num_nodes, num_edges))

    for i, (from_node, to_node, _, _) in enumerate(edges):
        A_eq[from_node, i] = 1  # Outgoing flow
        A_eq[to_node, i] = -1  # Incoming flow

    # b_eq vector (net supply at each node)
    # Positive = supply, Negative = demand
    b_eq = np.array([2500, 1000, 0, 0, -1000, -500, -2000])

    # Solve the linear program
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    print("=" * 60)
    print("Exercise 1.1: Initial Network Flow Problem")
    print("=" * 60)
    print(f"Status: {result.message}")
    print(f"Minimum cost: {result.fun:.2f}")
    print("\nOptimal flow:")
    for i, (from_node, to_node, cost, capacity) in enumerate(edges):
        node_names = [
            "Älmhult",
            "Markaryd",
            "Liatorp",
            "Osby",
            "Ljungby",
            "Alvesta",
            "Växjö",
        ]
        if result.x[i] > 0.001:  # Only show non-zero flows
            print(
                f"  {node_names[from_node]:10s} -> {node_names[to_node]:10s}: {result.x[i]:8.2f} units (cost: {cost}, capacity: {capacity})"
            )

    return result.fun, result.x


def solve_exercise_1_2_varnamo():
    """
    Part 2a: Evaluate Varnamo as new production facility
    """

    # Nodes: Älmhult(0), Markaryd(1), Varnamo(2), Liatorp(3), Osby(4), Ljungby(5), Alvesta(6), Växjö(7)

    # Updated edges with Varnamo
    edges = [
        (0, 3, 18, 1000),  # Älmhult -> Liatorp
        (0, 4, 24, 1000),  # Älmhult -> Osby
        (0, 7, 62, 1000),  # Älmhult -> Växjö
        (4, 1, 29, 1000),  # Osby -> Markaryd
        (3, 5, 31, 500),  # Liatorp -> Ljungby
        (3, 7, 46, 1000),  # Liatorp -> Växjö
        (1, 5, 51, 2000),  # Markaryd -> Ljungby
        (5, 6, 42, 500),  # Ljungby -> Alvesta
        (6, 7, 20, 2000),  # Alvesta -> Växjö
        (2, 5, 43, 2000),  # Varnamo -> Ljungby
        (2, 6, 50, 500),  # Varnamo -> Alvesta
    ]

    num_edges = len(edges)
    c = np.array([edge[2] for edge in edges])
    bounds = [(0, edge[3]) for edge in edges]

    # Updated supply/demand with Varnamo
    # Supply: Älmhult=2000, Markaryd=750, Varnamo=750
    # Demand: Ljungby=1000, Alvesta=500, Växjö=2000
    num_nodes = 8
    A_eq = np.zeros((num_nodes, num_edges))

    for i, (from_node, to_node, _, _) in enumerate(edges):
        A_eq[from_node, i] = 1
        A_eq[to_node, i] = -1

    b_eq = np.array([2000, 750, 750, 0, 0, -1000, -500, -2000])

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    print("\n" + "=" * 60)
    print("Exercise 1.2a: Network Flow with Varnamo")
    print("=" * 60)
    print(f"Status: {result.message}")
    print(f"Minimum cost: {result.fun:.2f}")
    print("\nOptimal flow:")
    for i, (from_node, to_node, cost, capacity) in enumerate(edges):
        node_names = [
            "Älmhult",
            "Markaryd",
            "Varnamo",
            "Liatorp",
            "Osby",
            "Ljungby",
            "Alvesta",
            "Växjö",
        ]
        if result.x[i] > 0.001:
            print(
                f"  {node_names[from_node]:10s} -> {node_names[to_node]:10s}: {result.x[i]:8.2f} units (cost: {cost}, capacity: {capacity})"
            )

    return result.fun, result.x


def solve_exercise_1_2_vislanda():
    """
    Part 2b: Evaluate Vislanda as new production facility
    """

    # Nodes: Älmhult(0), Markaryd(1), Vislanda(2), Liatorp(3), Osby(4), Ljungby(5), Alvesta(6), Växjö(7)

    edges = [
        (0, 3, 18, 1000),  # Älmhult -> Liatorp
        (0, 4, 24, 1000),  # Älmhult -> Osby
        (0, 7, 62, 1000),  # Älmhult -> Växjö
        (4, 1, 29, 1000),  # Osby -> Markaryd
        (3, 5, 31, 500),  # Liatorp -> Ljungby
        (3, 7, 46, 1000),  # Liatorp -> Växjö
        (1, 5, 51, 2000),  # Markaryd -> Ljungby
        (5, 6, 42, 500),  # Ljungby -> Alvesta
        (6, 7, 20, 2000),  # Alvesta -> Växjö
        (2, 6, 15, 500),  # Vislanda -> Alvesta
        (2, 7, 29, 500),  # Vislanda -> Växjö
    ]

    num_edges = len(edges)
    c = np.array([edge[2] for edge in edges])
    bounds = [(0, edge[3]) for edge in edges]

    # Supply: Älmhult=2120, Markaryd=1000, Vislanda=380
    # Demand: Ljungby=1000, Alvesta=500, Växjö=2000
    num_nodes = 8
    A_eq = np.zeros((num_nodes, num_edges))

    for i, (from_node, to_node, _, _) in enumerate(edges):
        A_eq[from_node, i] = 1
        A_eq[to_node, i] = -1

    b_eq = np.array([2120, 1000, 380, 0, 0, -1000, -500, -2000])

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    print("\n" + "=" * 60)
    print("Exercise 1.2b: Network Flow with Vislanda")
    print("=" * 60)
    print(f"Status: {result.message}")
    print(f"Minimum cost: {result.fun:.2f}")
    print("\nOptimal flow:")
    for i, (from_node, to_node, cost, capacity) in enumerate(edges):
        node_names = [
            "Älmhult",
            "Markaryd",
            "Vislanda",
            "Liatorp",
            "Osby",
            "Ljungby",
            "Alvesta",
            "Växjö",
        ]
        if result.x[i] > 0.001:
            print(
                f"  {node_names[from_node]:10s} -> {node_names[to_node]:10s}: {result.x[i]:8.2f} units (cost: {cost}, capacity: {capacity})"
            )

    return result.fun, result.x


if __name__ == "__main__":
    # Solve all parts
    cost_1, flow_1 = solve_exercise_1_1()
    cost_varnamo, flow_varnamo = solve_exercise_1_2_varnamo()
    cost_vislanda, flow_vislanda = solve_exercise_1_2_vislanda()

    # Compare results
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Initial network cost: {cost_1:.2f}")
    print(f"Varnamo network cost: {cost_varnamo:.2f}")
    print(f"Vislanda network cost: {cost_vislanda:.2f}")
    print(
        f"\nRecommendation: {'Vislanda' if cost_vislanda < cost_varnamo else 'Varnamo'} should be chosen"
    )
    print(f"Cost savings with best option: {abs(cost_vislanda - cost_varnamo):.2f}")

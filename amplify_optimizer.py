"""
Amplify: graph-based resource optimization examples for humanitarian nonprofits.

This script contains two standalone optimization models:

1. Volunteer scheduling on a bipartite graph
   - Nodes represent volunteers and shifts.
   - Edges represent assignments that are allowed by availability.
   - The optimizer selects assignments that maximize coverage and match quality.

2. Transport corridor selection on a logistics graph
   - Nodes represent depots and service locations.
   - Edges represent candidate transport links.
   - The optimizer selects links that fit a budget and reach high-need locations.

Both models are written so the graph data is easy to change in one place.
The code can be published on GitHub without requiring context from earlier files.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from qiskit_optimization import QuadraticProgram

warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class Volunteer:
    """A person available to help with one or more shifts."""

    name: str
    max_shifts: int


@dataclass(frozen=True)
class Shift:
    """A work period that needs a minimum and maximum number of volunteers."""

    name: str
    min_staff: int
    max_staff: int
    priority: int


@dataclass(frozen=True)
class AssignmentEdge:
    """An allowed volunteer-to-shift assignment with a preference score."""

    volunteer: str
    shift: str
    score: int


@dataclass(frozen=True)
class Location:
    """A logistics location with a demand score."""

    name: str
    demand: int
    is_hub: bool = False


@dataclass(frozen=True)
class RouteEdge:
    """A candidate transport link with cost and impact values."""

    start: str
    end: str
    cost: int
    reliability: int


VOLUNTEER_SCENARIO = {
    "volunteers": [
        Volunteer("Ava", max_shifts=2),
        Volunteer("Ben", max_shifts=1),
        Volunteer("Chloe", max_shifts=2),
        Volunteer("Diego", max_shifts=1),
        Volunteer("Ella", max_shifts=2),
    ],
    "shifts": [
        Shift("FoodBank-AM", min_staff=1, max_staff=2, priority=5),
        Shift("Shelter-PM", min_staff=2, max_staff=3, priority=6),
        Shift("Clinic-Evening", min_staff=1, max_staff=2, priority=7),
    ],
    "assignments": [
        AssignmentEdge("Ava", "FoodBank-AM", 9),
        AssignmentEdge("Ava", "Shelter-PM", 6),
        AssignmentEdge("Ben", "FoodBank-AM", 7),
        AssignmentEdge("Ben", "Clinic-Evening", 8),
        AssignmentEdge("Chloe", "Shelter-PM", 9),
        AssignmentEdge("Chloe", "Clinic-Evening", 7),
        AssignmentEdge("Diego", "Shelter-PM", 8),
        AssignmentEdge("Ella", "FoodBank-AM", 5),
        AssignmentEdge("Ella", "Shelter-PM", 7),
        AssignmentEdge("Ella", "Clinic-Evening", 9),
    ],
}


TRANSPORT_SCENARIO = {
    "locations": [
        Location("MainDepot", demand=0, is_hub=True),
        Location("NorthShelter", demand=8),
        Location("RiverClinic", demand=10),
        Location("FoodWarehouse", demand=6),
        Location("SchoolKitchen", demand=7),
        Location("WaterPoint", demand=9),
    ],
    "routes": [
        RouteEdge("MainDepot", "NorthShelter", cost=3, reliability=8),
        RouteEdge("MainDepot", "RiverClinic", cost=4, reliability=9),
        RouteEdge("MainDepot", "FoodWarehouse", cost=2, reliability=7),
        RouteEdge("FoodWarehouse", "SchoolKitchen", cost=3, reliability=8),
        RouteEdge("RiverClinic", "WaterPoint", cost=2, reliability=9),
        RouteEdge("NorthShelter", "SchoolKitchen", cost=2, reliability=6),
        RouteEdge("FoodWarehouse", "WaterPoint", cost=4, reliability=7),
        RouteEdge("SchoolKitchen", "WaterPoint", cost=2, reliability=8),
    ],
    "route_budget": 10,
}


@dataclass
class SolveSummary:
    """A small result container used by both classical and QAOA solvers."""

    objective_value: float
    variables_dict: Dict[str, int]


def build_volunteer_problem(
    volunteers: Sequence[Volunteer],
    shifts: Sequence[Shift],
    assignments: Sequence[AssignmentEdge],
) -> Tuple[QuadraticProgram, nx.Graph]:
    """
    Create a linear binary optimization model for volunteer scheduling.

    Decision variable x_volunteer_shift = 1 if that assignment is selected.
    """

    qp = QuadraticProgram("volunteer_scheduling")
    graph = nx.Graph()

    volunteer_lookup = {volunteer.name: volunteer for volunteer in volunteers}
    shift_lookup = {shift.name: shift for shift in shifts}

    for volunteer in volunteers:
        graph.add_node(volunteer.name, kind="volunteer")
    for shift in shifts:
        graph.add_node(shift.name, kind="shift")

    objective: Dict[str, float] = {}

    for edge in assignments:
        variable = assignment_var_name(edge.volunteer, edge.shift)
        qp.binary_var(variable)
        graph.add_edge(edge.volunteer, edge.shift, score=edge.score)

        # Higher-priority shifts are more valuable to cover.
        shift_value = shift_lookup[edge.shift].priority
        objective[variable] = edge.score + shift_value

    # Each volunteer can only work up to their capacity.
    for volunteer in volunteers:
        relevant_vars = [
            assignment_var_name(edge.volunteer, edge.shift)
            for edge in assignments
            if edge.volunteer == volunteer.name
        ]
        qp.linear_constraint(
            linear={var: 1 for var in relevant_vars},
            sense="<=",
            rhs=volunteer.max_shifts,
            name=f"volunteer_capacity_{volunteer.name}",
        )

    # Each shift must receive a minimum and maximum number of volunteers.
    for shift in shifts:
        relevant_vars = [
            assignment_var_name(edge.volunteer, edge.shift)
            for edge in assignments
            if edge.shift == shift.name
        ]
        qp.linear_constraint(
            linear={var: 1 for var in relevant_vars},
            sense=">=",
            rhs=shift.min_staff,
            name=f"shift_min_{shift.name}",
        )
        qp.linear_constraint(
            linear={var: 1 for var in relevant_vars},
            sense="<=",
            rhs=shift.max_staff,
            name=f"shift_max_{shift.name}",
        )

    qp.maximize(linear=objective)
    return qp, graph


def build_transport_problem(
    locations: Sequence[Location],
    routes: Sequence[RouteEdge],
    route_budget: int,
) -> Tuple[QuadraticProgram, nx.Graph]:
    """
    Create a binary optimization model for selecting transport links.

    The model chooses a subset of routes under a cost budget.
    A location counts as served when at least one selected route touches it,
    or when the location is already a hub.
    """

    qp = QuadraticProgram("transport_corridor_selection")
    graph = nx.Graph()

    location_lookup = {location.name: location for location in locations}

    for location in locations:
        graph.add_node(location.name, demand=location.demand, is_hub=location.is_hub)
        if not location.is_hub:
            qp.binary_var(served_var_name(location.name))

    route_cost_terms: Dict[str, float] = {}
    objective: Dict[str, float] = {}

    for route in routes:
        variable = route_var_name(route.start, route.end)
        qp.binary_var(variable)
        graph.add_edge(route.start, route.end, cost=route.cost, reliability=route.reliability)
        route_cost_terms[variable] = route.cost

        # Small negative cost pushes the model away from wasteful routes.
        objective[variable] = -0.35 * route.cost

    # Respect the overall transport budget.
    qp.linear_constraint(
        linear=route_cost_terms,
        sense="<=",
        rhs=route_budget,
        name="route_budget",
    )

    # A non-hub location can only be marked as served if a selected route touches it.
    for location in locations:
        if location.is_hub:
            continue

        served_var = served_var_name(location.name)
        incident_route_vars = [
            route_var_name(route.start, route.end)
            for route in routes
            if route.start == location.name or route.end == location.name
        ]

        qp.linear_constraint(
            linear={served_var: 1, **{var: -1 for var in incident_route_vars}},
            sense="<=",
            rhs=0,
            name=f"served_only_if_connected_{location.name}",
        )

        # Demand and reliability are the value we get from reaching a location.
        reliability_bonus = sum(
            route.reliability
            for route in routes
            if route.start == location.name or route.end == location.name
        ) / max(len(incident_route_vars), 1)
        objective[served_var] = location.demand + 0.2 * reliability_bonus

    # Encourage routes that start from a hub to make the network practical.
    for route in routes:
        start_hub = location_lookup[route.start].is_hub
        end_hub = location_lookup[route.end].is_hub
        if start_hub or end_hub:
            objective[route_var_name(route.start, route.end)] += 1.0

    qp.maximize(linear=objective)
    return qp, graph


def assignment_var_name(volunteer: str, shift: str) -> str:
    """Create a readable variable name for a volunteer assignment."""

    return f"assign_{volunteer}_to_{shift}"


def served_var_name(location: str) -> str:
    """Create the variable name used for whether a location is served."""

    return f"served_{location}"


def route_var_name(start: str, end: str) -> str:
    """Create a readable variable name for a route selection."""

    ordered = sorted((start, end))
    return f"route_{ordered[0]}__{ordered[1]}"


def solve_volunteer_classically(
    volunteers: Sequence[Volunteer],
    shifts: Sequence[Shift],
    assignments: Sequence[AssignmentEdge],
) -> SolveSummary:
    """
    Solve the volunteer model by checking every allowed assignment pattern.

    This is a practical baseline for small demos. It also makes the script
    usable even when someone wants a quick result without waiting for QAOA.
    """

    best_score = float("-inf")
    best_variables: Dict[str, int] = {}

    shift_lookup = {shift.name: shift for shift in shifts}

    for bit_values in product([0, 1], repeat=len(assignments)):
        volunteer_load = {volunteer.name: 0 for volunteer in volunteers}
        shift_load = {shift.name: 0 for shift in shifts}
        variables: Dict[str, int] = {}
        score = 0.0
        valid = True

        for edge, chosen in zip(assignments, bit_values):
            variable = assignment_var_name(edge.volunteer, edge.shift)
            variables[variable] = chosen
            if chosen == 0:
                continue

            volunteer_load[edge.volunteer] += 1
            shift_load[edge.shift] += 1
            score += edge.score + shift_lookup[edge.shift].priority

        for volunteer in volunteers:
            if volunteer_load[volunteer.name] > volunteer.max_shifts:
                valid = False
                break

        if not valid:
            continue

        for shift in shifts:
            assigned = shift_load[shift.name]
            if assigned < shift.min_staff or assigned > shift.max_staff:
                valid = False
                break

        if valid and score > best_score:
            best_score = score
            best_variables = variables

    return SolveSummary(objective_value=best_score, variables_dict=best_variables)


def solve_transport_classically(
    locations: Sequence[Location],
    routes: Sequence[RouteEdge],
    route_budget: int,
) -> SolveSummary:
    """
    Solve the transport model by checking each route subset under the budget.

    For a small publishable example this is easier to understand than using a
    heavyweight exact eigensolver, and it avoids the memory blow-up that would
    happen when the binary model grows.
    """

    location_lookup = {location.name: location for location in locations}
    best_score = float("-inf")
    best_variables: Dict[str, int] = {}

    for bit_values in product([0, 1], repeat=len(routes)):
        selected_routes = []
        cost = 0
        score = 0.0
        variables: Dict[str, int] = {}

        for route, chosen in zip(routes, bit_values):
            variable = route_var_name(route.start, route.end)
            variables[variable] = chosen
            if chosen == 0:
                continue

            selected_routes.append(route)
            cost += route.cost
            score += -0.35 * route.cost
            if location_lookup[route.start].is_hub or location_lookup[route.end].is_hub:
                score += 1.0

        if cost > route_budget:
            continue

        served_locations = set()
        for location in locations:
            if location.is_hub:
                served_locations.add(location.name)

        for route in selected_routes:
            served_locations.add(route.start)
            served_locations.add(route.end)

        for location in locations:
            if location.is_hub:
                continue

            served_var = served_var_name(location.name)
            served = int(location.name in served_locations)
            variables[served_var] = served
            if served:
                incident_routes = [
                    route
                    for route in routes
                    if route.start == location.name or route.end == location.name
                ]
                reliability_bonus = sum(route.reliability for route in incident_routes) / max(
                    len(incident_routes), 1
                )
                score += location.demand + 0.2 * reliability_bonus

        if score > best_score:
            best_score = score
            best_variables = variables

    return SolveSummary(objective_value=best_score, variables_dict=best_variables)


def solve_with_qaoa(qp: QuadraticProgram, reps: int, shots: int, maxiter: int) -> SolveSummary:
    """Solve a Qiskit optimization model with QAOA on the Aer simulator."""

    from qiskit import Aer
    from qiskit.algorithms import QAOA
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit.utils import QuantumInstance
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    backend = Aer.get_backend("qasm_simulator")
    quantum_instance = QuantumInstance(backend=backend, shots=shots)
    optimizer = COBYLA(maxiter=maxiter)
    qaoa = QAOA(optimizer=optimizer, reps=reps, quantum_instance=quantum_instance)
    result = MinimumEigenOptimizer(qaoa).solve(qp)
    return SolveSummary(
        objective_value=result.fval,
        variables_dict={name: int(round(value)) for name, value in result.variables_dict.items()},
    )


def report_volunteer_solution(
    graph: nx.Graph,
    assignments: Sequence[AssignmentEdge],
    result: SolveSummary,
) -> List[Tuple[str, str]]:
    """Turn the raw optimization result into a readable schedule."""

    chosen_assignments: List[Tuple[str, str]] = []
    print("\nVolunteer schedule")
    print("-" * 40)

    for edge in assignments:
        variable = assignment_var_name(edge.volunteer, edge.shift)
        value = result.variables_dict.get(variable, 0)
        if round(value) == 1:
            chosen_assignments.append((edge.volunteer, edge.shift))
            print(f"{edge.volunteer:>6} -> {edge.shift:<16} score={edge.score}")

    if not chosen_assignments:
        print("No assignments were selected.")

    return chosen_assignments


def report_transport_solution(
    graph: nx.Graph,
    locations: Sequence[Location],
    routes: Sequence[RouteEdge],
    result: SolveSummary,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Turn the raw optimization result into selected routes and served locations."""

    chosen_routes: List[Tuple[str, str]] = []
    served_locations: List[str] = []

    print("\nTransport plan")
    print("-" * 40)

    for route in routes:
        variable = route_var_name(route.start, route.end)
        value = result.variables_dict.get(variable, 0)
        if round(value) == 1:
            chosen_routes.append((route.start, route.end))
            print(f"{route.start:>14} <-> {route.end:<14} cost={route.cost}")

    for location in locations:
        if location.is_hub:
            served_locations.append(location.name)
            continue

        variable = served_var_name(location.name)
        value = result.variables_dict.get(variable, 0)
        if round(value) == 1:
            served_locations.append(location.name)

    print("\nReached locations:", ", ".join(served_locations))
    return chosen_routes, served_locations


def draw_volunteer_graph(
    graph: nx.Graph,
    chosen_assignments: Iterable[Tuple[str, str]],
    output_path: str,
) -> None:
    """Save a picture of the volunteer graph with selected assignments highlighted."""

    chosen_set = {tuple(edge) for edge in chosen_assignments}
    volunteers = [node for node, data in graph.nodes(data=True) if data["kind"] == "volunteer"]
    shifts = [node for node, data in graph.nodes(data=True) if data["kind"] == "shift"]

    pos = {}
    for index, volunteer in enumerate(volunteers):
        pos[volunteer] = (0, -index)
    for index, shift in enumerate(shifts):
        pos[shift] = (2.5, -index)

    edge_colors = [
        "#198754" if (u, v) in chosen_set or (v, u) in chosen_set else "#b0b7c3"
        for u, v in graph.edges()
    ]
    edge_widths = [3.2 if color == "#198754" else 1.4 for color in edge_colors]
    node_colors = ["#7cc6fe" if node in volunteers else "#ffd166" for node in graph.nodes()]

    plt.figure(figsize=(10, 6))
    nx.draw(
        graph,
        pos=pos,
        with_labels=True,
        node_color=node_colors,
        node_size=2400,
        edge_color=edge_colors,
        width=edge_widths,
        font_size=10,
    )
    edge_labels = {(u, v): data["score"] for u, v, data in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=9)
    plt.title("Amplify volunteer scheduling graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def draw_transport_graph(
    graph: nx.Graph,
    chosen_routes: Iterable[Tuple[str, str]],
    served_locations: Iterable[str],
    output_path: str,
) -> None:
    """Save a picture of the logistics graph with selected routes highlighted."""

    chosen_route_set = {tuple(sorted(edge)) for edge in chosen_routes}
    served_set = set(served_locations)

    pos = nx.spring_layout(graph, seed=7)
    edge_colors = [
        "#198754" if tuple(sorted((u, v))) in chosen_route_set else "#adb5bd"
        for u, v in graph.edges()
    ]
    edge_widths = [3.2 if color == "#198754" else 1.6 for color in edge_colors]

    node_colors = []
    for node, data in graph.nodes(data=True):
        if data.get("is_hub"):
            node_colors.append("#ef476f")
        elif node in served_set:
            node_colors.append("#06d6a0")
        else:
            node_colors.append("#8d99ae")

    plt.figure(figsize=(9, 7))
    nx.draw(
        graph,
        pos=pos,
        with_labels=True,
        node_color=node_colors,
        node_size=2200,
        edge_color=edge_colors,
        width=edge_widths,
        font_size=9,
    )
    edge_labels = {(u, v): data["cost"] for u, v, data in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=9)
    plt.title("Amplify transport corridor graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_volunteer_demo(solver_name: str, reps: int, shots: int, maxiter: int, output_path: str) -> None:
    """Run the volunteer scheduling example end to end."""

    qp, graph = build_volunteer_problem(
        volunteers=VOLUNTEER_SCENARIO["volunteers"],
        shifts=VOLUNTEER_SCENARIO["shifts"],
        assignments=VOLUNTEER_SCENARIO["assignments"],
    )

    if solver_name == "classical":
        result = solve_volunteer_classically(
            volunteers=VOLUNTEER_SCENARIO["volunteers"],
            shifts=VOLUNTEER_SCENARIO["shifts"],
            assignments=VOLUNTEER_SCENARIO["assignments"],
        )
    else:
        result = solve_with_qaoa(qp=qp, reps=reps, shots=shots, maxiter=maxiter)

    print(f"\nSolver: {solver_name}")
    print(f"Objective value: {result.objective_value:.3f}")
    chosen_assignments = report_volunteer_solution(
        graph=graph,
        assignments=VOLUNTEER_SCENARIO["assignments"],
        result=result,
    )
    draw_volunteer_graph(graph=graph, chosen_assignments=chosen_assignments, output_path=output_path)
    print(f"\nSaved graph image to {output_path}")


def run_transport_demo(solver_name: str, reps: int, shots: int, maxiter: int, output_path: str) -> None:
    """Run the transport corridor example end to end."""

    qp, graph = build_transport_problem(
        locations=TRANSPORT_SCENARIO["locations"],
        routes=TRANSPORT_SCENARIO["routes"],
        route_budget=TRANSPORT_SCENARIO["route_budget"],
    )

    if solver_name == "classical":
        result = solve_transport_classically(
            locations=TRANSPORT_SCENARIO["locations"],
            routes=TRANSPORT_SCENARIO["routes"],
            route_budget=TRANSPORT_SCENARIO["route_budget"],
        )
    else:
        result = solve_with_qaoa(qp=qp, reps=reps, shots=shots, maxiter=maxiter)

    print(f"\nSolver: {solver_name}")
    print(f"Objective value: {result.objective_value:.3f}")
    chosen_routes, served_locations = report_transport_solution(
        graph=graph,
        locations=TRANSPORT_SCENARIO["locations"],
        routes=TRANSPORT_SCENARIO["routes"],
        result=result,
    )
    draw_transport_graph(
        graph=graph,
        chosen_routes=chosen_routes,
        served_locations=served_locations,
        output_path=output_path,
    )
    print(f"\nSaved graph image to {output_path}")


def parse_args() -> argparse.Namespace:
    """Define command-line options so the script is easy to reuse."""

    parser = argparse.ArgumentParser(
        description="Amplify graph optimization examples for humanitarian nonprofits."
    )
    parser.add_argument(
        "--scenario",
        choices=("volunteers", "transport"),
        default="volunteers",
        help="Choose which graph-based optimization example to run.",
    )
    parser.add_argument(
        "--solver",
        choices=("classical", "qaoa"),
        default="classical",
        help="Use the small-instance classical baseline or the QAOA workflow.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=2,
        help="Depth of the QAOA circuit when --solver qaoa is used.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Number of simulator shots when --solver qaoa is used.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=150,
        help="Maximum number of COBYLA iterations when --solver qaoa is used.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="amplify_solution.png",
        help="Path for the saved graph image.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the selected Amplify scenario."""

    args = parse_args()

    if args.scenario == "volunteers":
        run_volunteer_demo(
            solver_name=args.solver,
            reps=args.reps,
            shots=args.shots,
            maxiter=args.maxiter,
            output_path=args.output,
        )
        return

    run_transport_demo(
        solver_name=args.solver,
        reps=args.reps,
        shots=args.shots,
        maxiter=args.maxiter,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

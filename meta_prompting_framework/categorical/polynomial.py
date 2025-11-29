"""
Polynomial Functor abstraction for meta-prompting framework.

Implements Spivak's polynomial functors for bidirectional tool/agent composition.
"""

from typing import TypeVar, Generic, Callable, List, Tuple, Any, Dict
from dataclasses import dataclass

Position = TypeVar('Position')
Direction = TypeVar('Direction')
P = TypeVar('P')
D = TypeVar('D')


@dataclass
class PolynomialFunctor(Generic[Position, Direction]):
    """
    A polynomial functor p = Σᵢ y^(Aᵢ)

    Models bidirectional interaction:
    - Positions: Output states (the indices i)
    - Directions: Input requests at each position (the sets Aᵢ)

    Example (tool/agent):
    - Position: Current state after tool execution
    - Direction: What inputs the tool needs at that state

    This captures the "interface" of a tool:
    - Forward pass: Execute tool, get position (result state)
    - Backward pass: At each position, what directions (inputs) are needed?

    Based on Spivak's work on polynomial functors and dynamical systems.
    """

    positions: List[Position]
    directions: Callable[[Position], List[Direction]]

    def map_position(
        self,
        f: Callable[[Position], P]
    ) -> 'PolynomialFunctor[P, Direction]':
        """
        Map over positions (covariant).

        Transforms output states while preserving input structure.
        """
        return PolynomialFunctor(
            positions=[f(p) for p in self.positions],
            directions=lambda p_new: self.directions(
                # Find original position that mapped to p_new
                next(orig_p for orig_p in self.positions if f(orig_p) == p_new)
            )
        )

    def map_direction(
        self,
        f: Callable[[Direction], D]
    ) -> 'PolynomialFunctor[Position, D]':
        """
        Map over directions (contravariant).

        Transforms input requests while preserving output structure.
        """
        return PolynomialFunctor(
            positions=self.positions,
            directions=lambda p: [f(d) for d in self.directions(p)]
        )

    def compose(
        self,
        other: 'PolynomialFunctor'
    ) -> 'PolynomialFunctor':
        """
        Polynomial composition p ◁ q.

        Output of q becomes input to p.
        This is the fundamental operation for composing tools/agents.

        The composition creates a new polynomial where:
        - Positions are pairs (position from p, position from q)
        - Directions combine directions from both
        """
        new_positions = [
            (p_pos, q_pos)
            for p_pos in self.positions
            for q_pos in other.positions
        ]

        def new_directions(pos_pair):
            p_pos, q_pos = pos_pair
            # Combine directions from both polynomials
            p_dirs = self.directions(p_pos)
            q_dirs = other.directions(q_pos)
            return [(p_dir, q_dir) for p_dir in p_dirs for q_dir in q_dirs]

        return PolynomialFunctor(new_positions, new_directions)

    def __repr__(self):
        return f"Poly({len(self.positions)} positions)"


@dataclass
class Lens(Generic[Position, Direction]):
    """
    A lens is a special polynomial functor y^A.

    Represents a single position with A possible directions.
    Fundamental building block for tool interfaces.

    A lens encodes:
    - get: Extract a value from a structure
    - set: Update a value in a structure

    Example (database lens):
    - get: Query current state
    - set: Update state with new value
    """

    get: Callable[[Position], Direction]  # Forward: extract value
    set: Callable[[Position, Direction], Position]  # Backward: update

    def to_polynomial(self) -> PolynomialFunctor[Position, Direction]:
        """Convert lens to polynomial functor representation."""
        # A lens has one position and extracts one direction
        def directions(pos: Position) -> List[Direction]:
            return [self.get(pos)]

        return PolynomialFunctor(
            positions=[],  # Position is external (the structure being focused on)
            directions=directions
        )

    def compose(self, other: 'Lens') -> 'Lens':
        """
        Lens composition (sequential access).

        If lens1 focuses on A in S, and lens2 focuses on B in A,
        then lens1.compose(lens2) focuses on B in S.
        """
        return Lens(
            get=lambda s: other.get(self.get(s)),
            set=lambda s, b: self.set(s, other.set(self.get(s), b))
        )

    def __repr__(self):
        return f"Lens(get={self.get.__name__}, set={self.set.__name__})"


class ToolInterface:
    """
    Model a tool (e.g., database query, API call) as a polynomial functor.

    Positions: Query results or states
    Directions: Needed parameters for each result

    Example (database tool):
    - Position "empty": needs {"table": str, "columns": List[str]}
    - Position "results": needs {"next_page": int, "filters": dict}
    - Position "complete": needs nothing
    """

    def __init__(self, tool_name: str, state_machine: Dict[str, List[Dict]]):
        """
        Initialize tool interface.

        Args:
            tool_name: Name of the tool
            state_machine: Mapping from positions to required directions
        """
        self.tool_name = tool_name
        self.state_machine = state_machine

    def as_polynomial(self) -> PolynomialFunctor[str, Dict]:
        """
        Represent tool as polynomial functor.

        Returns:
            Polynomial functor encoding tool interface
        """
        positions = list(self.state_machine.keys())

        def directions(position: str) -> List[Dict]:
            return self.state_machine.get(position, [])

        return PolynomialFunctor(positions, directions)

    def __repr__(self):
        return f"Tool({self.tool_name})"


def wire_tools(
    tool1: PolynomialFunctor,
    tool2: PolynomialFunctor
) -> PolynomialFunctor:
    """
    Compose tools using polynomial composition.

    Output of tool1 feeds into input of tool2.

    This creates a "wiring diagram" showing how data flows
    through the composed system.

    Args:
        tool1: First tool
        tool2: Second tool

    Returns:
        Composed polynomial functor
    """
    return tool2.compose(tool1)


def test_polynomial_functors():
    """
    Test polynomial functor functionality.

    Returns:
        Dictionary of test results
    """
    results = {}

    # Test basic polynomial functor
    positions = ["state_a", "state_b"]
    directions_func = lambda pos: ["input1", "input2"] if pos == "state_a" else ["input3"]

    poly = PolynomialFunctor(positions, directions_func)

    results["positions_stored"] = poly.positions == positions
    results["directions_callable"] = poly.directions("state_a") == ["input1", "input2"]

    # Test map_position (covariant)
    poly_mapped = poly.map_position(lambda p: p.upper())
    results["map_position_transforms"] = "STATE_A" in poly_mapped.positions

    # Test map_direction (contravariant)
    poly_dir_mapped = poly.map_direction(lambda d: d + "_transformed")
    results["map_direction_transforms"] = poly_dir_mapped.directions("state_a")[0] == "input1_transformed"

    # Test lens
    # Example: Focus on "name" field in a dictionary
    def get_name(d: Dict) -> str:
        return d.get("name", "")

    def set_name(d: Dict, name: str) -> Dict:
        return {**d, "name": name}

    name_lens = Lens(get_name, set_name)

    person = {"name": "Alice", "age": 30}
    results["lens_get"] = name_lens.get(person) == "Alice"

    updated = name_lens.set(person, "Bob")
    results["lens_set"] = updated["name"] == "Bob" and updated["age"] == 30

    # Test lens composition
    # Lens 1: Focus on "address" in person
    # Lens 2: Focus on "city" in address

    def get_address(d: Dict) -> Dict:
        return d.get("address", {})

    def set_address(d: Dict, addr: Dict) -> Dict:
        return {**d, "address": addr}

    def get_city(d: Dict) -> str:
        return d.get("city", "")

    def set_city(d: Dict, city: str) -> Dict:
        return {**d, "city": city}

    address_lens = Lens(get_address, set_address)
    city_lens = Lens(get_city, set_city)

    # Compose: person -> address -> city
    city_in_person_lens = address_lens.compose(city_lens)

    person_with_address = {
        "name": "Alice",
        "address": {"city": "NYC", "zip": "10001"}
    }

    results["lens_composition_get"] = city_in_person_lens.get(person_with_address) == "NYC"

    updated_person = city_in_person_lens.set(person_with_address, "LA")
    results["lens_composition_set"] = updated_person["address"]["city"] == "LA"

    # Test ToolInterface
    db_tool = ToolInterface(
        "database",
        {
            "empty": [{"table": str, "columns": List[str]}],
            "results": [{"next_page": int}],
            "complete": []
        }
    )

    db_poly = db_tool.as_polynomial()
    results["tool_interface_positions"] = "empty" in db_poly.positions
    results["tool_interface_directions"] = len(db_poly.directions("empty")) == 1

    return results


if __name__ == "__main__":
    results = test_polynomial_functors()

    print("Polynomial Functor Test Results:")
    print("=" * 50)
    for test_name, passed in results.items():
        print(f"  {test_name}: {'✓' if passed else '✗'}")

    print("\nAll tests passed!" if all(results.values()) else "\nSome tests failed!")

    # Demonstration
    print("\n" + "=" * 50)
    print("Demonstration: Tool Composition with Polynomial Functors")
    print("=" * 50)

    # Create two tools
    retrieval_tool = ToolInterface(
        "retrieval",
        {
            "idle": [{"query": str}],
            "searching": [{"max_results": int}],
            "complete": []
        }
    )

    ranking_tool = ToolInterface(
        "ranking",
        {
            "idle": [{"documents": List[str]}],
            "ranking": [{"criteria": str}],
            "complete": []
        }
    )

    print(f"\nTool 1: {retrieval_tool}")
    print(f"  States: {list(retrieval_tool.state_machine.keys())}")

    print(f"\nTool 2: {ranking_tool}")
    print(f"  States: {list(ranking_tool.state_machine.keys())}")

    # Compose tools
    composed = wire_tools(
        retrieval_tool.as_polynomial(),
        ranking_tool.as_polynomial()
    )

    print(f"\nComposed tool:")
    print(f"  Positions (state pairs): {len(composed.positions)}")
    print(f"  This represents all possible state combinations of the two tools")

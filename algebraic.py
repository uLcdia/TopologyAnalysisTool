import numpy as np
from typing import Tuple, Optional, List

class AlgebraicStructure:
    """Analyzes algebraic properties of a computer network represented as a latency matrix.
    
    The class determines if a network forms a valid algebraic structure based on the
    bridge computer concept, where each pair of computers must communicate through
    an intermediate bridge computer that provides the optimal indirect path.

    Key Properties:
    1. Well-defined: Each pair (A,B) must have exactly one optimal bridge computer C
    2. Closure: Bridge computers must exist within the network
    3. Indirect paths: Direct-only connections are invalid (security requirement)
    """

    def __init__(self, matrix: np.ndarray):
        """Initialize with a latency matrix where -1 indicates no connection."""
        self.matrix = matrix
        self.n_computers = len(matrix)
        self._bridge_computers = {}
        self._is_algebraic_structure = None
        self._algebraic_structure_message = None
        self._algebraic_structure_invalid_pairs = None

    def find_bridge_computer(self, a: int, b: int) -> Tuple[Optional[int], float]:
        """Find the optimal bridge computer between computers a and b.
        
        Returns:
            Tuple[Optional[int], float]: (bridge_computer_id, total_distance)
            where bridge_computer_id is None if no valid bridge exists
        """
        cache_key = (a, b)
        if cache_key in self._bridge_computers:
            return self._bridge_computers[cache_key]

        best_distance = float('inf')
        best_bridge = None

        for c in range(self.n_computers):
            if (self.matrix[a][c] != -1 and
                self.matrix[c][b] != -1 and
                c != a and c != b):
                total_distance = self.matrix[a][c] + self.matrix[c][b]
                if total_distance < best_distance:
                    best_distance = total_distance
                    best_bridge = c

        result = (best_bridge, best_distance)
        self._bridge_computers[cache_key] = result
        return result

    def is_algebraic_structure(self) -> Tuple[bool, List[Tuple[int, int]]]:
        """Determine if the network forms a valid algebraic structure.
        
        Returns:
            Tuple[bool, List[Tuple[int, int]]]: (is_valid, invalid_pairs)
            where invalid_pairs lists computer pairs without valid bridge computers
        """
        if self._is_algebraic_structure is not None:
            return self._is_algebraic_structure, self._algebraic_structure_invalid_pairs

        invalid_pairs = []
        # Check each pair of computers for valid bridge computers
        for a in range(self.n_computers):
            for b in range(self.n_computers):
                if a != b:
                    bridge, _ = self.find_bridge_computer(a, b)
                    if bridge is None:
                        invalid_pairs.append((a, b))

        # Cache and return results
        self._is_algebraic_structure = len(invalid_pairs) == 0
        self._algebraic_structure_invalid_pairs = invalid_pairs
        return self._is_algebraic_structure, self._algebraic_structure_invalid_pairs

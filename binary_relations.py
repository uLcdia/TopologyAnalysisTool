import numpy as np
import pandas as pd

class BinaryRelations:
    """Analyzes binary relations in a computer network represented as a matrix.
    
    The class converts a latency matrix to a binary accessibility matrix and
    provides methods to analyze its mathematical properties (reflexivity,
    symmetry, transitivity) and compute closures.
    """

    def __init__(self, matrix: np.ndarray):
        """Initialize with a latency matrix where -1 indicates no connection."""
        self.accessibility_matrix = (matrix != -1).astype(int)
        self.n_computers = len(matrix)
        
        # Cache for computed properties
        self._is_reflexive = None
        self._is_symmetric = None
        self._is_transitive = None
        self._symmetric_closure = None
        self._transitive_closure = None

    def is_reflexive(self) -> bool:
        """Check if the relation is reflexive: (a,a) ∈ R for all a."""
        if self._is_reflexive is None:
            self._is_reflexive = all(self.accessibility_matrix[i][i] == 1 for i in range(self.n_computers))
        return self._is_reflexive

    def is_symmetric(self) -> bool:
        """Check if the relation is symmetric: if (a,b) ∈ R then (b,a) ∈ R."""
        if self._is_symmetric is None:
            self._is_symmetric = np.array_equal(self.accessibility_matrix, self.accessibility_matrix.T)
        return self._is_symmetric

    def is_transitive(self) -> bool:
        """Check if the relation is transitive: if (a,b) ∈ R and (b,c) ∈ R then (a,c) ∈ R."""
        if self._is_transitive is None:
            matrix_product = np.dot(self.accessibility_matrix, self.accessibility_matrix)
            self._is_transitive = np.all(np.logical_or(matrix_product == 0, self.accessibility_matrix > 0))
        return self._is_transitive

    def get_symmetric_closure(self) -> np.ndarray:
        """Compute the smallest symmetric relation containing R."""
        if self._symmetric_closure is None:
            self._symmetric_closure = np.maximum(self.accessibility_matrix, self.accessibility_matrix.T)
        return self._symmetric_closure

    def get_transitive_closure(self) -> np.ndarray:
        """Compute the smallest transitive relation containing R using Warshall's algorithm."""
        if self._transitive_closure is None:
            closure = self.accessibility_matrix.copy()
            for k in range(self.n_computers):
                for i in range(self.n_computers):
                    for j in range(self.n_computers):
                        closure[i][j] = closure[i][j] or (closure[i][k] and closure[k][j])
            self._transitive_closure = closure
        return self._transitive_closure

    def _create_styled_df(self, matrix: np.ndarray) -> pd.DataFrame:
        # Common DataFrame creation and styling logic
        df = pd.DataFrame(
            matrix,
            index=[f'PC {i}' for i in range(self.n_computers)],
            columns=[f'PC {i}' for i in range(self.n_computers)]
        )
        return df.style.map(
            lambda x: 'background-color: lightgray' if x == 0 else None
        ).format("{:.0f}")

    def get_symmetric_closure_df(self) -> pd.DataFrame:
        return self._create_styled_df(self.get_symmetric_closure())

    def get_transitive_closure_df(self) -> pd.DataFrame:
        return self._create_styled_df(self.get_transitive_closure())
    
    def get_accessibility_matrix_df(self) -> pd.DataFrame:
        return self._create_styled_df(self.get_accessibility_matrix())

    def get_relation_properties(self) -> dict:
        properties = {
            'reflexive': {
                'status': self.is_reflexive()
            },
            'symmetric': {
                'status': self.is_symmetric()
            },
            'transitive': {
                'status': self.is_transitive()
            }
        }
        return properties

    def get_accessibility_matrix(self) -> np.ndarray:
        return self.accessibility_matrix
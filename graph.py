import numpy as np
from typing import List, Tuple, Set
import heapq

class GraphTheory:
    """Analyzes directed weighted graphs in a computer network represented as a matrix.
    
    The class analyzes graph properties and computes fundamental graph algorithms on a
    latency matrix where:
    - Edge weights represent latencies between computers
    - -1 indicates no connection
    - 0 indicates self-connection
    
    Mathematical properties analyzed:
    - Path existence and shortest paths (Dijkstra's algorithm)
    - Strong connectivity (complete reachability between all vertices)
    - Eulerian properties (existence of a circuit using each edge exactly once)
    - Hamiltonian properties (existence of a cycle visiting each vertex exactly once)
    - Minimum spanning trees (minimum total weight tree reaching all vertices)
    """

    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
        self.n_computers = len(matrix)
        self._shortest_paths = {}
        self._accessibility_matrix = None
        self._is_strongly_connected = None
        self._degrees = None
        self._is_eulerian = None
        self._is_hamiltonian = None
        self._minimum_spanning_tree = {}

    # Dijkstra's Algorithm
    # Purpose: Finds the shortest path between two nodes in a weighted graph
    # Time Complexity: O(E log V) where E is edges and V is vertices
    def get_shortest_path(self, start: int, end: int) -> Tuple[float, List[int]]:
        """Find shortest path between two nodes using Dijkstra's algorithm.
        
        Uses a min-priority queue implementation with time complexity O(E log V).
        Returns (path, distance) where path is empty and distance is inf if no path exists.
        """
        cache_key = (start, end)
        if cache_key in self._shortest_paths:
            return self._shortest_paths[cache_key]

        distances = [float('inf')] * self.n_computers
        distances[start] = 0
        predecessors = [-1] * self.n_computers
        pq = [(0, start)]
        visited = set()

        while pq:
            current_distance, current = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            if current == end:
                break

            for next_node in range(self.n_computers):
                if self.matrix[current][next_node] != -1:
                    distance = current_distance + self.matrix[current][next_node]
                    if distance < distances[next_node]:
                        distances[next_node] = distance
                        predecessors[next_node] = current
                        heapq.heappush(pq, (distance, next_node))

        # Reconstruct path
        path = []
        current = end
        while current != -1:
            path.append(current)
            current = predecessors[current]

        result = (path[::-1] if distances[end] != float('inf') else [], distances[end])
        self._shortest_paths[cache_key] = result
        return result

    # Floyd-Warshall-like Accessibility Matrix
    # Purpose: Creates a binary matrix showing which nodes can reach other nodes
    # Time Complexity: O(V^3) due to running Dijkstra's for each pair of vertices
    def get_accessibility_matrix(self) -> np.ndarray:
        """Create binary matrix showing which nodes can reach other nodes.
        
        Element (i,j) is 1 if there exists a path from node i to node j, 0 otherwise.
        """
        if self._accessibility_matrix is not None:
            return self._accessibility_matrix

        accessibility = np.zeros((self.n_computers, self.n_computers), dtype=int)

        for i in range(self.n_computers):
            for j in range(self.n_computers):
                distance, path = self.get_shortest_path(i, j)
                accessibility[i][j] = 1 if distance != float('inf') else 0

        self._accessibility_matrix = accessibility
        return accessibility

    # Strong Connectivity Check
    # Purpose: Verifies if every node can reach every other node
    # Uses Dijkstra's algorithm for each pair of vertices
    # Time Complexity: O(V^3)
    def is_strongly_connected(self) -> bool:
        """Check if graph is strongly connected.
        
        A directed graph is strongly connected if there exists a path from any vertex
        to any other vertex.
        """
        if self._is_strongly_connected is not None:
            return self._is_strongly_connected

        for i in range(self.n_computers):
            for j in range(self.n_computers):
                if i != j:
                    _, path = self.get_shortest_path(i, j)
                    if not path:
                        self._is_strongly_connected = False
                        return False
        
        self._is_strongly_connected = True
        return True

    # Degree Calculator
    # Purpose: Calculates in-degree and out-degree for each vertex
    # Time Complexity: O(V^2)
    def get_degrees(self):
        """Calculate in-degree and out-degree for each vertex.
        
        Returns tuple of (in_degrees, out_degrees) lists.
        """
        if self._degrees is not None:
            return self._degrees

        in_degrees = []
        out_degrees = []

        for i in range(self.n_computers):
            in_deg = sum(1 for j in range(self.n_computers)
                        if self.matrix[j][i] not in [-1, 0])
            out_deg = sum(1 for j in range(self.n_computers)
                         if self.matrix[i][j] not in [-1, 0])
            in_degrees.append(in_deg)
            out_degrees.append(out_deg)

        self._degrees = (in_degrees, out_degrees)
        return self._degrees

    # Eulerian Circuit Check
    # Purpose: Determines if graph has an Eulerian circuit
    # Uses strong connectivity check and degree properties
    # Time Complexity: O(V^3)
    def is_eulerian(self) -> Tuple[bool, str]:
        """Check if graph contains an Eulerian circuit.
        
        A directed graph has an Eulerian circuit if and only if:
        1. All vertices with nonzero degree belong to a single strongly connected component
        2. For every vertex, in-degree equals out-degree
        
        Returns (is_eulerian, explanation).
        """
        if self._is_eulerian is not None:
            return self._is_eulerian

        if not self.is_strongly_connected():
            result = (False, "Not strongly connected")
            self._is_eulerian = result
            return result

        in_degrees, out_degrees = self.get_degrees()

        # For directed graph to be Eulerian:
        # 1. All vertices with nonzero degree belong to a single strongly connected component
        # 2. For every vertex, in-degree equals out-degree
        for in_deg, out_deg in zip(in_degrees, out_degrees):
            if in_deg != out_deg:
                result = (False, "In-degrees and out-degrees are not equal for all vertices")
                self._is_eulerian = result
                return result

        result = (True, "")
        self._is_eulerian = result
        return result

    # Hamiltonian Cycle Detection
    # Purpose: Determines if graph has a Hamiltonian cycle
    # Uses backtracking algorithm
    # Time Complexity: O(n!)
    def _hamiltonian_util(self, path: List[int], pos: int, visited: Set[int]) -> bool:
        # If all vertices are visited, check if there's an edge back to start
        if len(visited) == self.n_computers:
            return self.matrix[path[-1]][path[0]] not in [-1, 0]

        # Try different vertices as next candidates
        for v in range(self.n_computers):
            # If vertex not visited and there's an edge to it
            if v not in visited and self.matrix[path[pos-1]][v] not in [-1, 0]:
                path[pos] = v
                visited.add(v)

                if self._hamiltonian_util(path, pos + 1, visited):
                    return True

                # Backtrack
                path[pos] = -1
                visited.remove(v)

        return False

    def is_hamiltonian(self) -> Tuple[bool, str]:
        """Check if graph contains a Hamiltonian cycle using backtracking.
        
        A Hamiltonian cycle visits each vertex exactly once and returns to start.
        NP-complete problem solved using backtracking.
        
        Returns (is_hamiltonian, explanation).
        """
        if self._is_hamiltonian is not None:
            return self._is_hamiltonian

        if not self.is_strongly_connected():
            return False, "Not strongly connected"
        
        # Check if any vertex is isolated
        in_degrees, out_degrees = self.get_degrees()
        if any(in_deg == 0 or out_deg == 0 for in_deg, out_deg in zip(in_degrees, out_degrees)):
            return False, "Graph contains isolated vertices"
        
        # Initialize path with -1
        path = [-1] * self.n_computers
        path[0] = 0  # Start from vertex 0
        visited = {0}

        result = (True, "") if self._hamiltonian_util(path, 1, visited) else (False, "No Hamiltonian path found")
        self._is_hamiltonian = result
        return result

    # Minimum Spanning Tree (Modified Chu-Liu/Edmonds' Algorithm)
    # Purpose: Finds the minimum spanning tree in a directed graph
    # Uses cycle contraction and recursive approach
    # Time Complexity: O(VE)
    def get_minimum_spanning_tree(self, start: int) -> Tuple[List[Tuple[int, int]], float]:
        """Find minimum spanning tree using modified Chu-Liu/Edmonds' algorithm.
        
        For directed graphs, finds the minimum spanning arborescence (directed tree)
        rooted at the start vertex that reaches all other vertices with minimum total weight.
        
        Returns (edges, total_weight) where edges is list of (from, to) tuples.
        """
        if start in self._minimum_spanning_tree:
            return self._minimum_spanning_tree[start]

        if not self.is_strongly_connected():
            return [], float('inf')

        # Initialize the minimum incoming edges for each vertex
        min_edges = {}
        total_distance = 0

        # Find minimum incoming edge for each vertex except start
        for v in range(self.n_computers):
            if v == start:
                continue
            min_distance = float('inf')
            min_source = -1
            for u in range(self.n_computers):
                if self.matrix[u][v] not in [-1, 0] and self.matrix[u][v] < min_distance:
                    min_distance = self.matrix[u][v]
                    min_source = u
            if min_source == -1:
                return [], float('inf')
            min_edges[v] = (min_source, v, min_distance)
            total_distance += min_distance

        # Check for cycles
        visited = set()
        cycles = []
        cycle_vertices = set()

        def find_cycle(vertex, path):
            if vertex in path:
                cycle_start = path.index(vertex)
                cycles.append(path[cycle_start:])
                cycle_vertices.update(path[cycle_start:])
                return
            if vertex in visited:
                return
            visited.add(vertex)
            if vertex != start and vertex in min_edges:
                path.append(vertex)
                find_cycle(min_edges[vertex][0], path)
                path.pop()

        # Find cycles in the graph
        for v in range(self.n_computers):
            if v not in visited and v != start:
                find_cycle(v, [])

        # If no cycles, we're done
        if not cycles:
            result = ([(edge[0], edge[1]) for edge in min_edges.values()], total_distance)
            self._minimum_spanning_tree[start] = result
            return result

        # Contract cycles and recursively solve
        for cycle in cycles:
            # Find minimum incoming edge to cycle
            min_external_distance = float('inf')
            min_external_edge = None
            cycle_set = set(cycle)
            
            for v in cycle:
                for u in range(self.n_computers):
                    if u not in cycle_set and self.matrix[u][v] not in [-1, 0]:
                        adj_weight = self.matrix[u][v] - (min_edges[v][2] if v in min_edges else 0)
                        if adj_weight < min_external_distance:
                            min_external_distance = adj_weight
                            min_external_edge = (u, v)

            if min_external_edge:
                # Replace the existing edge with the minimum external edge
                old_edge = min_edges[min_external_edge[1]]
                min_edges[min_external_edge[1]] = (min_external_edge[0], min_external_edge[1], 
                                                 self.matrix[min_external_edge[0]][min_external_edge[1]])
                total_distance = total_distance - old_edge[2] + min_edges[min_external_edge[1]][2]

        result = ([(edge[0], edge[1]) for edge in min_edges.values()], total_distance)
        self._minimum_spanning_tree[start] = result
        return result
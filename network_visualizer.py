import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

class NetworkVisualizer:
    """Visualizes directed networks with optional path and MST highlighting.
    
    Provides consistent network visualization with automatic sizing and coloring
    based on network size. Supports highlighting shortest paths and minimum
    spanning trees.
    """

    def __init__(self):
        self.node_positions = {}  # Cached node positions for consistent layout

    def get_drawing_params(self, n_nodes):
        """Return visualization parameters scaled to network size.
        
        Args:
            n_nodes: Number of nodes in the network
            
        Returns:
            dict: Visual parameters including sizes for nodes, fonts, and arrows
        """
        # Return visualization parameters based on network size
        # Smaller networks: larger nodes, fonts, and arrows
        # Larger networks: smaller elements for better spacing
        if n_nodes <= 5:
            return {'fig_size': 8, 'node_size': 700, 'font_size': 12, 
                    'arrow_size': 20, 'edge_width': 1.5}
        elif n_nodes <= 10:
            return {'fig_size': 9.6, 'node_size': 500, 'font_size': 10, 
                    'arrow_size': 15, 'edge_width': 1.2}
        else:
            return {'fig_size': 12, 'node_size': 300, 'font_size': 8, 
                    'arrow_size': 10, 'edge_width': 1}

    def draw_network(self, matrix, path=None, mst_edges=None, show_edge_labels=False):
        """Draw the network visualization.
        
        Args:
            matrix: Adjacency matrix representing the network
            path: List of node indices representing a path to highlight
            mst_edges: List of (source, target) tuples for MST edges
            show_edge_labels: Whether to show weight labels on highlighted edges
            
        Returns:
            matplotlib.figure.Figure: The generated network visualization
        """
        # Main method to visualize the network
        # - Creates directed graph from adjacency matrix
        # - Handles highlighting of paths and MST edges
        # - Returns matplotlib figure
        # Clear any existing plots
        plt.clf()
        plt.close('all')

        # Get parameters based on matrix size
        n_nodes = len(matrix)
        params = self.get_drawing_params(n_nodes)
        
        # Create a new figure with specific DPI
        fig = plt.figure(figsize=(params['fig_size'], params['fig_size']), dpi=100)
        
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(n_nodes):
            G.add_node(i)

        # Generate positions if they don't exist or node count changed
        if not self.node_positions or len(self.node_positions) != n_nodes:
            self.node_positions = nx.circular_layout(G)

        # Generate distinct colors
        colors = plt.cm.Set3(np.linspace(0, 1, n_nodes))
        node_colors_dict = {i: colors[i] for i in range(n_nodes)}

        # Modify the path handling for propagation view
        if mst_edges and not path:
            # Create a path from the MST edges for proper coloring
            connected_nodes = set()
            for edge in mst_edges:
                connected_nodes.add(edge[0])
                connected_nodes.add(edge[1])
            path = list(connected_nodes)  # Convert to list for node coloring

        self._draw_edges(G, matrix, n_nodes, params, path, mst_edges, show_edge_labels, node_colors_dict)
        self._draw_nodes(G, params, path, node_colors_dict)

        plt.margins(0.15)
        plt.tight_layout()
        
        plt.close('all')
        return fig

    def _draw_edges(self, G, matrix, n_nodes, params, path, mst_edges, show_edge_labels, node_colors_dict):
        """Draw all network edges with appropriate styling."""
        for i in range(n_nodes):
            for j in range(n_nodes):
                if matrix[i][j] not in [-1, 0] and i != j:
                    edge_color = node_colors_dict[i]
                    width = params['edge_width']

                    # Modified edge highlighting logic
                    is_path_edge = path and len(path) > 1 and any(
                        path[k] == i and path[k+1] == j 
                        for k in range(len(path)-1)
                    )
                    is_mst_edge = mst_edges and ((i,j) in mst_edges or (j,i) in mst_edges)

                    # Use different styling for MST edges vs path edges
                    if is_mst_edge:
                        edge_color = '#39a275'  # Use a consistent color for MST edges
                        width = params['edge_width'] * 2
                    elif is_path_edge:
                        edge_color = mcolors.rgb_to_hsv(edge_color[:3])
                        edge_color[1] *= 1.5
                        edge_color[2] *= 0.5
                        edge_color = mcolors.hsv_to_rgb(edge_color)
                        width = params['edge_width'] * 2

                    self._draw_single_edge(G, i, j, matrix, params, edge_color, width, 
                                         show_edge_labels, is_path_edge, is_mst_edge)

    def _draw_single_edge(self, G, i, j, matrix, params, edge_color, width, 
                         show_edge_labels, is_path_edge, is_mst_edge):
        """Draw a single edge with curved arrow and optional label."""
        # Draw individual edge with curved arrows
        # - Adds weight labels for path/MST edges if enabled
        rad = 0.2
        nx.draw_networkx_edges(G, self.node_positions,
                             edgelist=[(i,j)],
                             edge_color=[edge_color],
                             width=width,
                             arrowsize=params['arrow_size'],
                             arrowstyle='->',
                             connectionstyle=f'arc3, rad={rad}')

        if show_edge_labels and (is_path_edge or is_mst_edge):
            edge_labels = {(i,j): f'{matrix[i][j]}'}
            nx.draw_networkx_edge_labels(G, self.node_positions,
                                       edge_labels=edge_labels,
                                       label_pos=0.5,
                                       font_size=params['font_size'],
                                       rotate=True,
                                       bbox=dict(facecolor='white', 
                                               edgecolor='none', 
                                               alpha=0.3, 
                                               pad=0.5),
                                       connectionstyle=f'arc3, rad={rad}')

    def _draw_nodes(self, G, params, path, node_colors_dict):
        """Draw network nodes with special coloring for path endpoints."""
        # Draw network nodes
        # - Start node: red
        # - End node: green
        # - Other nodes: default colors
        node_colors = []
        for node in G.nodes():
            if path and len(path) > 1:
                if node == path[0]:
                    node_colors.append('#df1c44')
                elif node == path[-1]:
                    node_colors.append('#39a275')
                else:
                    node_colors.append(node_colors_dict[node])
            else:
                node_colors.append(node_colors_dict[node])

        nx.draw_networkx_nodes(G, self.node_positions,
                             node_color=node_colors,
                             node_size=params['node_size'],
                             edgecolors='white',
                             linewidths=params['edge_width'])

        nx.draw_networkx_labels(G, self.node_positions, font_size=params['font_size'])
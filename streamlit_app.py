import streamlit as st
import pandas as pd
from generate_matrix import generate_adjacency_matrix, save_matrix_to_csv, load_matrix_from_csv
from graph import GraphTheory
from algebraic import AlgebraicStructure
from binary_relations import BinaryRelations
from network_visualizer import NetworkVisualizer

node_positions = {}

def display_matrix_with_tags(matrix):
    """Convert adjacency matrix to styled pandas DataFrame with PC labels."""
    if matrix is None:
        return None
        
    df = pd.DataFrame(
        matrix,
        index=[f'PC {i}' for i in range(len(matrix))],
        columns=[f'PC {i}' for i in range(len(matrix))]
    ).style.map(
        lambda x: 'background-color: lightgray' if x == -1 else None
    ).format("{:.0f}")

    return df

def generate_matrix(n_computers, connectivity_prob, min_distance, max_distance):
    """Generate a new network adjacency matrix with given parameters."""
    st.session_state.matrix = generate_adjacency_matrix(
        n_computers,
        connectivity_prob,
        min_distance,
        max_distance
    )
    return st.session_state.matrix

def perform_network_structure_analysis():
    """Analyze network properties including binary relations, graph theory, and algebraic structure."""
    if not st.session_state.network_structure_results:
        # Perform basic analysis that doesn't depend on start/end nodes
        graph = st.session_state.graph
        alg = st.session_state.alg
        bin_rel = st.session_state.bin_rel
        
        network_structure_results = {
            'binary_relations': bin_rel.get_relation_properties(),
            'graph_properties': {
                'eulerian': graph.is_eulerian(),
                'hamiltonian': graph.is_hamiltonian(),
                'degrees': graph.get_degrees()
            },
            'algebraic_structure': alg.is_algebraic_structure()
        }
        st.session_state.network_structure_results = network_structure_results
    return st.session_state.network_structure_results

def perform_path_analysis(start, end):
    """Calculate shortest path, bridge computer, and propagation path between start and end nodes."""
    # Return cached results if unchanged
    if (st.session_state.path_analysis_cache and 
        st.session_state.path_analysis_cache['start'] == start and 
        st.session_state.path_analysis_cache['end'] == end):
        return st.session_state.path_analysis_cache

    # Cache new path analysis results
    st.session_state.path_analysis_cache = {
        'start': start,
        'end': end,
        'path': st.session_state.graph.get_shortest_path(start, end),
        'bridge': st.session_state.alg.find_bridge_computer(start, end)
    }

    # Handle shortest path visualization
    shortest_path, shortest_distance = st.session_state.path_analysis_cache['path']
    if shortest_distance != float('inf'):
        shortest_fig = st.session_state.visualizer.draw_network(st.session_state.matrix, path=shortest_path, show_edge_labels=True)
        st.session_state.shortest_path_results = {
            'start': start, 'end': end, 'path': shortest_path,
            'distance': shortest_distance, 'fig': shortest_fig
        }
    else:
        st.session_state.shortest_path_results = None

    # Handle bridge computer visualization
    bridge, bridge_distance = st.session_state.path_analysis_cache['bridge']
    if bridge is not None:
        bridge_path = [start, bridge, end]
        bridge_fig = st.session_state.visualizer.draw_network(st.session_state.matrix, path=bridge_path, show_edge_labels=True)
        st.session_state.bridge_results = {
            'start': start, 'end': end, 'bridge': bridge,
            'distance': bridge_distance, 'path': bridge_path, 'fig': bridge_fig
        }
    else:
        st.session_state.bridge_results = None

    # Handle propagation path visualization
    edges, propagation_distance = st.session_state.graph.get_minimum_spanning_tree(start)
    if edges:
        propagation_fig = st.session_state.visualizer.draw_network(st.session_state.matrix, path=[], mst_edges=edges, show_edge_labels=True)
        st.session_state.propagation_results = {
            'start': start, 'edges': edges,
            'distance': propagation_distance, 'fig': propagation_fig
        }
    else:
        st.session_state.propagation_results = None

    # Update visualization based on current view
    if st.session_state.current_view_tag == 'shortest_path' and st.session_state.shortest_path_results:
        st.session_state.fig = st.session_state.shortest_path_results['fig']
    elif st.session_state.current_view_tag == 'bridge' and st.session_state.bridge_results:
        st.session_state.fig = st.session_state.bridge_results['fig']
    elif st.session_state.current_view_tag == 'propagation' and st.session_state.propagation_results:
        st.session_state.fig = st.session_state.propagation_results['fig']
    else:
        st.session_state.current_view_tag = 'default'
        st.session_state.fig = st.session_state.visualizer.draw_network(st.session_state.matrix)

    return st.session_state.path_analysis_cache

def visualization():
    """Display the current network visualization using matplotlib."""
    if st.session_state.matrix is None:
        st.warning("No matrix available. Please generate a new matrix.")
        return
        
    # Use st.empty() to create a placeholder for the plot
    if 'plot_placeholder' not in st.session_state:
        st.session_state.plot_placeholder = st.empty()
    
    # Draw the network if there's no figure
    if st.session_state.fig is None:
        st.session_state.fig = st.session_state.visualizer.draw_network(st.session_state.matrix)
    
    # Always show the current figure
    st.session_state.plot_placeholder.pyplot(st.session_state.fig)

def store_matrix(matrix):
    """Store matrix in session state and initialize analysis objects."""
    st.session_state.matrix = matrix
    # Initialize analysis objects
    st.session_state.graph = GraphTheory(matrix)
    st.session_state.alg = AlgebraicStructure(matrix)
    st.session_state.bin_rel = BinaryRelations(matrix)
    st.session_state.visualizer = NetworkVisualizer()
    # Clear cached analysis results
    st.session_state.network_structure_results = None
    st.session_state.path_analysis_cache = None
    st.session_state.shortest_path_results = None
    st.session_state.bridge_results = None
    st.session_state.propagation_results = None
    # Clear cached figure
    st.session_state.fig = None

def main():
    """Main Streamlit application for network topology analysis tool."""
    st.title("Network Topology Analysis Tool")

    # Initialize session state
    if 'matrix' not in st.session_state:
        st.session_state.matrix = None
        st.session_state.graph = None
        st.session_state.alg = None
        st.session_state.bin_rel = None
        st.session_state.network_structure_results = None
        st.session_state.path_analysis_cache = None
        st.session_state.shortest_path_results = None
        st.session_state.bridge_results = None
        st.session_state.propagation_results = None
        st.session_state.fig = None
        st.session_state.current_view_tag = 'default'

    # Sidebar controls
    st.sidebar.subheader("Network Generation Parameters")
    gen_n_computers = st.sidebar.number_input("Number of computers", min_value=2, value=5)
    connectivity_prob = st.sidebar.slider("Connectivity Probability", 0.0, 1.0, 0.3)

    # Latency range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        min_distance = st.number_input("Min Latency", min_value=1, value=1)
    with col2:
        max_distance = st.number_input("Max Latency", min_value=min_distance, value=100)

    # Generate initial matrix if none exists
    if st.session_state.matrix is None:
        store_matrix(
            generate_matrix(
                gen_n_computers,
                connectivity_prob,
                min_distance,
                max_distance
            )
        )

    # Network generation
    if st.sidebar.button("Generate New Network"):
        store_matrix(
            generate_matrix(
                gen_n_computers,
                connectivity_prob,
                min_distance,
                max_distance
            )
        )
        save_matrix_to_csv(st.session_state.matrix, "network_matrix.csv")

    # Network Matrix import/export
    st.sidebar.subheader("Network Matrix Import/Export")
    uploaded_file = st.sidebar.file_uploader("Import Network Matrix", type=['csv'])
    if uploaded_file is not None:
        try:
            matrix = load_matrix_from_csv(uploaded_file)
            if matrix.shape[0] == matrix.shape[1]:  # Verify square matrix
                store_matrix(matrix)
                save_matrix_to_csv(matrix, "network_matrix.csv")
                st.sidebar.success("Network imported successfully!")
            else:
                st.sidebar.error("Imported file must be a square matrix!")
        except Exception as e:
            st.sidebar.error(f"Error importing file: {str(e)}")

    # Download button
    if st.sidebar.button("Export Network Matrix"):
        try:
            save_matrix_to_csv(st.session_state.matrix, "network_matrix.csv")
            with open("network_matrix.csv", 'rb') as f:
                st.sidebar.download_button(
                    label="Download CSV",
                    data=f,
                    file_name="network_matrix.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.sidebar.error(f"Error exporting file: {str(e)}")

    n_computers = len(st.session_state.matrix) if st.session_state.matrix is not None else 0

    # Display matrix
    st.subheader("Uplink Latency")
    if st.session_state.matrix is not None:
        st.write(display_matrix_with_tags(st.session_state.matrix))
    else:
        st.warning("No matrix available. Please generate a new matrix.")

    # Path finding
    st.subheader("Path Analysis")
    col1, col2 = st.columns(2)
    with col1:
        start = st.number_input("Start computer", min_value=0, max_value=n_computers-1)
    with col2:
        end = st.number_input("End computer", min_value=0, max_value=n_computers-1, value=n_computers-1)
            
    # Perform path analysis before view selection
    perform_path_analysis(start, end)

    # Display network graph (main visualization)
    st.subheader("Network Visualization")

    # Define view mappings (tag to display name and availability condition)
    view_mappings = {
        'default': {'name': 'Default View', 'condition': True},
        'shortest_path': {'name': 'Shortest Path', 'condition': bool(st.session_state.shortest_path_results)},
        'bridge': {'name': 'Bridge Path', 'condition': bool(st.session_state.bridge_results)},
        'propagation': {'name': 'Propagation Path', 'condition': bool(st.session_state.propagation_results)}
    }

    # Get available views
    available_view_tags = [tag for tag, info in view_mappings.items() if info['condition']]
    view_options = [view_mappings[tag]['name'] for tag in available_view_tags]
    
    # Reset to default view if current view is unavailable
    if st.session_state.current_view_tag not in available_view_tags:
        st.session_state.current_view_tag = 'default'
    
    try:
        current_index = available_view_tags.index(st.session_state.current_view_tag)
    except ValueError:
        current_index = 0
    
    # Create a callback for the radio button
    def on_view_change():
        try:
            selected_name = st.session_state.view_selector
            selected_index = view_options.index(selected_name)
            st.session_state.current_view_tag = available_view_tags[selected_index]
        except (ValueError, IndexError):
            st.session_state.current_view_tag = 'default'
    
    # Display radio buttons with the callback
    selected_name = st.radio(
        "Select View", 
        view_options, 
        index=current_index,
        key="view_selector",
        on_change=on_view_change
    )
    
    # Update figure based on current view tag
    if st.session_state.current_view_tag == 'shortest_path':
        st.session_state.fig = st.session_state.shortest_path_results['fig']
    elif st.session_state.current_view_tag == 'bridge':
        st.session_state.fig = st.session_state.bridge_results['fig']
    elif st.session_state.current_view_tag == 'propagation':
        st.session_state.fig = st.session_state.propagation_results['fig']
    else:
        st.session_state.fig = st.session_state.visualizer.draw_network(st.session_state.matrix)

    # Container for visualization
    viz_container = st.container()
    with viz_container:
        visualization()

    # Path finding results display
    path_col1, path_col2, path_col3 = st.columns(3)
    with path_col1:
        st.write("**Shortest Path**")
        if not st.session_state.shortest_path_results:
            st.error("No shortest path results available")
        else:
            st.success(f"Shortest path: {st.session_state.shortest_path_results['path']}")
            st.info(f"Total latency: {st.session_state.shortest_path_results['distance']}")

    with path_col2:
        st.write("**Bridge Computer**")
        if not st.session_state.bridge_results:
            st.error("No bridge computer results available")
        else:
            st.success(f"Bridge computer: {st.session_state.bridge_results['bridge']}")
            st.info(f"Total latency: {st.session_state.bridge_results['distance']}")

    with path_col3:
        st.write("**Propagation Path**")
        if not st.session_state.propagation_results:
            st.error("No propagation path results available")
        else:
            st.success(f"Propagation sequence: {st.session_state.propagation_results['edges']}")
            st.info(f"Total latency: {st.session_state.propagation_results['distance']}")

    st.session_state.plot_placeholder.pyplot(st.session_state.fig)

    # Combined Network Analysis
    st.subheader("Network Structure Analysis")
    perform_network_structure_analysis()
    network_col1, network_col2, network_col3 = st.columns(3)
    with network_col1:
        st.write("**Binary Relations**")
        properties = st.session_state.network_structure_results['binary_relations']
        for prop_name, prop_data in properties.items():
            if prop_data['status']:
                st.success(f"✓ {prop_name.title()}")
            else:
                st.error(f"✗ {prop_name.title()}")

    with network_col2:
        st.write("**Graph Properties**")
        graph_properties = st.session_state.network_structure_results['graph_properties']
        is_eulerian, euler_explanation = graph_properties['eulerian']
        is_hamiltonian, ham_explanation = graph_properties['hamiltonian']
        
        if is_eulerian:
            st.success("✓ Eulerian")
        else:
            st.error("✗ Not Eulerian")
            
        if is_hamiltonian:
            st.success("✓ Hamiltonian")
        else:
            st.error("✗ Not Hamiltonian")

    with network_col3:
        st.write("**Algebraic Structure**")
        is_algebraic, invalid_pairs = st.session_state.network_structure_results['algebraic_structure']
        if is_algebraic:
            st.success("✓ Valid")
        else:
            st.error("✗ Invalid")
    
    # Show detailed analysis in expandable sections
    with st.expander("Show Detailed Analysis"):
        relation_col1, relation_col2 = st.columns(2)
        with relation_col1:
            st.write("**Direct Accessibility:**")
            st.write(st.session_state.bin_rel.get_accessibility_matrix_df())
        with relation_col2:
            st.write("**Node Degrees:**")
            in_degrees, out_degrees = graph_properties['degrees']
            degree_df = pd.DataFrame({
                'In-Degree': in_degrees,
                'Out-Degree': out_degrees
            })
            st.dataframe(degree_df)

        if euler_explanation:
            st.write("**Eulerian:**")
            st.warning(euler_explanation)
        if ham_explanation:
            st.write("**Hamiltonian:**")
            st.warning(ham_explanation)

        closure_col1, closure_col2 = st.columns(2)
        with closure_col1:
            st.write("**Symmetric Closure:**")
            st.write(st.session_state.bin_rel.get_symmetric_closure_df())
        with closure_col2:
            st.write("**Transitive Closure:**")
            st.write(st.session_state.bin_rel.get_transitive_closure_df())

        if not is_algebraic and invalid_pairs:
            st.write("**Invalid Algebraic Pairs:**")
            for a, b in invalid_pairs:
                direct_distance = st.session_state.matrix[a][b]
                direct_str = f"Direct connection: {direct_distance}" if direct_distance != -1 else "No direct connection"
                bridge, distance = st.session_state.alg.find_bridge_computer(a, b)
                bridge_str = f"Best bridge: PC {bridge} (latency: {distance})" if bridge is not None else "No valid bridge computer"
                st.write(f"PC {a} → PC {b}:")
                st.write(f"- {direct_str}")
                st.write(f"- {bridge_str}")

if __name__ == "__main__":
    main()
# Network Topology Analysis Tool

A network topology analysis tool that visualizes and analyzes computer networks using graph theory, algebraic structures, and binary relations.

This is a project for the course "Discrete Mathematics".

## Overview

This project provides a comprehensive analysis of computer networks by modeling them as directed weighted graphs. Each edge represents a network connection with an associated latency, allowing for detailed analysis of network properties and optimal routing paths.

## Network Properties

### Connection Rules
- Networks consist of n computers with directed connections
- Each connection has an associated latency (weight)
- Self-connections have 0 latency
- Missing connections are represented as -1
- Connections are asymmetric (A → B ≠ B → A)

### Analysis Features

#### Graph Theory
- **Shortest Path**: Uses Dijkstra's algorithm to find optimal routes between computers
- **Network Propagation**: Implements Chu-Liu/Edmonds' algorithm for minimum spanning arborescence
- **Circuit Analysis**: Determines Eulerian and Hamiltonian properties
- **Accessibility**: Generates reachability matrix showing possible connections

#### Algebraic Structure
- **Bridge Computer Analysis**: Identifies optimal intermediate nodes for indirect routing
- **Circuit Validation**: Ensures data transmission follows security requirements
- **Structure Verification**: Validates algebraic properties of the network topology

#### Binary Relations
- **Direct Accessibility**: Maps immediate connections between computers
- **Relation Properties**: 
  - Reflexivity: Self-connections (A → A)
  - Symmetry: Bidirectional connections (A → B ⇔ B → A)
  - Transitivity: Connection chains (A → B → C ⇒ A → C)
- **Closure Analysis**: Computes symmetric and transitive closures

## Mathematical Foundation

### Graph Theory
- Directed weighted graphs
- Path optimization algorithms
- Circuit existence theorems

### Algebraic Structures
- Function mapping: f(A, B) = C where C is the optimal bridge computer
- Structure validation through closure and well-defined operations

### Binary Relations
- Set theory application to computer connections
- Relation properties and closures
- Matrix representations of network relationships

## Requirements
- Computer
- Python 3.10+
- Browser
- Ability to read

## Usage
- Clone the repository: `git clone https://github.com/uLcdia/TopologyAnalysisTool.git && cd TopologyAnalysisTool`
- Create a virtual environment: `python -m venv venv`
- Activate the virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Run the tool: `streamlit run streamlit_app.py`

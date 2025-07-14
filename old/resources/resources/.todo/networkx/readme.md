# NetworkX Cheatsheet

## 1. Installing NetworkX
- pip install networkx  # Install NetworkX

## 2. Importing Libraries
- import networkx as nx  # Import NetworkX library

## 3. Creating a Graph
- G = nx.Graph()  # Create an empty graph
- G = nx.DiGraph()  # Create a directed graph

## 4. Adding Nodes
- G.add_node(1)  # Add a single node
- G.add_nodes_from([2, 3, 4])  # Add multiple nodes

## 5. Adding Edges
- G.add_edge(1, 2)  # Add a single edge
- G.add_edges_from([(1, 2), (2, 3)])  # Add multiple edges

## 6. Drawing the Graph
- import matplotlib.pyplot as plt  # Import Matplotlib
- nx.draw(G, with_labels=True)  # Draw the graph with labels
- plt.show()  # Show plot

## 7. Analyzing the Graph
- num_nodes = G.number_of_nodes()  # Get the number of nodes
- num_edges = G.number_of_edges()  # Get the number of edges
- neighbors = list(G.neighbors(1))  # Get neighbors of a node

## 8. Graph Properties
- is_connected = nx.is_connected(G)  # Check if the graph is connected
- degree = G.degree(1)  # Get the degree of a node

## 9. Shortest Path
- shortest_path = nx.shortest_path(G, source=1, target=3)  # Get the shortest path
- path_length = nx.shortest_path_length(G, source=1, target=3)  # Get path length

## 10. Centrality Measures
- degree_centrality = nx.degree_centrality(G)  # Calculate degree centrality
- closeness_centrality = nx.closeness_centrality(G)  # Calculate closeness centrality
- betweenness_centrality = nx.betweenness_centrality(G)  # Calculate betweenness centrality

## 11. Saving and Loading Graphs
- nx.write_gml(G, 'graph.gml')  # Save graph to GML format
- G_loaded = nx.read_gml('graph.gml')  # Load graph from GML format

## 12. Working with Weighted Graphs
- G.add_edge(1, 2, weight=4.5)  # Add edge with weight
- weight = G[1][2]['weight']  # Access the weight of an edge
- shortest_path_weighted = nx.shortest_path(G, source=1, target=3, weight='weight')  # Shortest path with weights

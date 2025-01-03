import streamlit as st
from pyvis.network import Network
import networkx as nx
import tempfile
from pathlib import Path
import json
from graph import GraphRAG

def create_interactive_graph(G):
    # Convert NetworkX graph to Pyvis
    net = Network(height="750px", width="100%", bgcolor="#ffffff", 
                 font_color="black")
    
    # Copy node data
    for node, node_data in G.nodes(data=True):
        net.add_node(node, 
                    title=f"Type: {node_data.get('type', 'Unknown')}\n"
                          f"Description: {node_data.get('description', 'None')}")
    
    # Copy edge data
    for source, target, edge_data in G.edges(data=True):
        net.add_edge(source, target, 
                    title=edge_data.get('relationship', 'Unknown'),
                    value=edge_data.get('strength', 1))
    
    return net

def main():
    st.title("GraphRAG Interactive Visualization")
    
    # Initialize GraphRAG
    graph_rag = GraphRAG()
    
    # Load or generate graph
    if st.button("Load/Refresh Graph"):
        with st.spinner("Processing documents..."):
            documents = graph_rag.read_documents("data/")
            text_chunks = graph_rag.text_chunks(documents)
            element_instances = graph_rag.element_instances(text_chunks)
            
            # Create interactive visualization
            net = create_interactive_graph(element_instances)
            
            # Save and display
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                net.save_graph(tmp.name)
                with open(tmp.name, 'r', encoding='utf-8') as f:
                    html = f.read()
                st.components.v1.html(html, height=800)
            
            # Add filters
            st.sidebar.header("Filters")
            
            # Node type filter
            node_types = set(nx.get_node_attributes(element_instances, 'type').values())
            selected_types = st.sidebar.multiselect("Filter by node type", 
                                                  list(node_types))
            
            # Edge weight filter
            min_weight = st.sidebar.slider("Minimum edge weight", 0, 10, 0)
            
            if selected_types or min_weight > 0:
                filtered_graph = element_instances.copy()
                if selected_types:
                    filtered_graph = filtered_graph.subgraph(
                        [n for n, d in filtered_graph.nodes(data=True) 
                         if d.get('type') in selected_types]
                    )
                if min_weight > 0:
                    filtered_graph = nx.Graph(
                        (u, v, d) for u, v, d in filtered_graph.edges(data=True) 
                        if d.get('strength', 0) >= min_weight
                    )
                
                net = create_interactive_graph(filtered_graph)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                    net.save_graph(tmp.name)
                    with open(tmp.name, 'r', encoding='utf-8') as f:
                        html = f.read()
                    st.components.v1.html(html, height=800)

if __name__ == "__main__":
    main()
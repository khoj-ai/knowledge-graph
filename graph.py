"""
This file will contain a simple version of Graph RAG, which can be used to index and perform RAG on datasets, providing greater meta-level information about the dataset. This can help LLMs to better understand the dataset and improve their performance. The overall process of constructing the graph is as follows:
  1. Source Documents -> Text Chunks
  2. Text Chunks -> Element Instances
     - Create the graph representation. Extract nodes, edges via an LLM for entity extraction.
  3. Element Instances -> Element Summaries
     - Create a summary of each node
  4. Element Summaries -> Graph Communities
  5. Graph Communities -> Community Summaries
"""

import os
from typing import Dict, List
import json
import pickle
import collections
from prompts_templates import GRAPH_EXTRACTION_JSON_PROMPT, GRAPH_EXTRACTION_SYSTEM_PROMPT, NODE_SUMMARIZATION_SYSTEM_PROMPT, NODE_SUMMARIZATION_PROMPT, COMMUNITY_SUMMARIZATION_SYSTEM_PROMPT, COMMUNITY_SUMMARIZATION_PROMPT
from utils import clean_json
from operations import create_summary, run_leiden
from dotenv import load_dotenv
import logging
from tqdm import tqdm
import time

import networkx as nx

import google.generativeai as genai # type: ignore

from google.generativeai.types.safety_types import ( # type: ignore
    HarmBlockThreshold,
    HarmCategory,
) 

load_dotenv()

TEXT_CHUNK_SIZE = 600

N_CHUNKS_IN_ENTITY_BATCH = 10

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphRAG:
    def __init__(self):
        # Initialize the graph structure
        self.entity_extractor_model = genai.GenerativeModel(
            "gemini-1.5-flash-002",
            system_instruction=GRAPH_EXTRACTION_SYSTEM_PROMPT,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        self.summarizer_model = genai.GenerativeModel(
            "gemini-1.5-flash-002",
            system_instruction=NODE_SUMMARIZATION_SYSTEM_PROMPT,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        self.community_summarizer_model = genai.GenerativeModel(
            "gemini-1.5-flash-002",
            system_instruction=COMMUNITY_SUMMARIZATION_SYSTEM_PROMPT,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            }
        )

    def read_documents(self, directory) -> Dict[str, str]:
        # Read documents from the specified directory
        documents = dict()
        for filename in os.listdir(directory):
            with open(os.path.join(directory, filename), "r") as file:
                documents[filename] = file.read()
        return documents

    def text_chunks(self, documents: Dict[str, str]) -> Dict[str, List[str]]:
        # Convert source documents to text chunks
        text_chunks: Dict[str, List[str]] = dict()
        for document in documents:
            text_chunks[document] = []
            for i in range(0, len(documents[document]), TEXT_CHUNK_SIZE):
                text_chunks[document].append(documents[document][i:i+TEXT_CHUNK_SIZE])
        return text_chunks

    def element_instances(self, text_chunks):
        # Extract nodes and edges to create the graph representation
        graph_filepath = "graphs/entities.gpickle"
        
        try:
            if os.path.exists(graph_filepath):
                pickled_graph = pickle.load(open(graph_filepath, "rb"))
                return pickled_graph
        except:
            logger.info("No pickled graph found. Generating a new graph.")
            pass

        graph = nx.Graph()
        start_time = time.time()
        
        for document in tqdm(text_chunks, desc="Processing documents"):
            chunks = text_chunks[document]
            for i in tqdm(range(0, len(chunks), N_CHUNKS_IN_ENTITY_BATCH), desc=f"Processing chunk batches for {document}"):
                batch = chunks[i:i + N_CHUNKS_IN_ENTITY_BATCH]
                combined_text = " ".join(batch)
                prompt = GRAPH_EXTRACTION_JSON_PROMPT.format(
                    entity_types="Person, Organization, Location",
                    input_text=combined_text,
                    language="English"
                )
                chat_session = self.entity_extractor_model.start_chat()
                response = chat_session.send_message(prompt)
                try:
                    entities_and_relationships = json.loads(clean_json(response.text))
                except json.JSONDecodeError:
                    logger.error("Error parsing JSON response")
                    continue
                except Exception as e:
                    logger.error(f"Error: {e}", exc_info=True)
                    continue
                
                for item in entities_and_relationships:
                    if "name" in item:
                        graph.add_node(
                            item.get("name", "Unknown"),
                            type=item.get("type", "Unknown"),
                            description=item.get("description", "No description provided")
                        )
                    elif "source" in item:
                        graph.add_edge(
                            item.get("source", "Unknown"),
                            item.get("target", "Unknown"),
                            relationship=item.get("relationship", "Unknown"),
                            strength=item.get("relationship_strength", 1)
                        )
        end_time = time.time()
        logger.info(f"Processed all documents in {end_time - start_time:.2f} seconds")
        pickle.dump(graph, open(graph_filepath, "wb"))
        return graph

    def element_summaries(self, element_instances: nx.Graph) -> nx.Graph:
        # Create a summary of each node
        summary_graph = nx.DiGraph()
        
        node_summary_graph_file_path = "graphs/node_summaries.gpickle"
        try:
            if os.path.exists(node_summary_graph_file_path):
                pickled_graph = pickle.load(open(node_summary_graph_file_path, "rb"))
                return pickled_graph
        except:
            logger.info("No pickled graph found. Generating a new graph.")
            pass
        
        for node, data in element_instances.nodes(data=True):
            all_edges = list(element_instances.edges(node, data=True))
            summary_details = {
                "name": node,
                "type": data.get("type", "Unknown"),
                "description": data.get("description", "No description provided"),
                "edges": [
                    {
                        "target": edge[1],
                        "relationship": edge[2].get("relationship", "Unknown"),
                        "strength": edge[2].get("strength", 1)
                    }
                    for edge in all_edges
                ]
            }
            chat_session = self.summarizer_model.start_chat()
            formatted_message = NODE_SUMMARIZATION_PROMPT.format(
                node_name=summary_details["name"],
                node_type=summary_details["type"],
                node_description=summary_details["description"],
                relationships=summary_details["edges"]
            )
            
            response = chat_session.send_message(formatted_message)
            
            summary_graph.add_node(node, summary=response.text if response.text else "No summary provided")
            
        pickle.dump(summary_graph, open(node_summary_graph_file_path, "wb"))
        
        return summary_graph

    def graph_communities(self, element_summaries: nx.Graph) -> nx.DiGraph:
        # Group nodes into communities
        community_map_to_cluster, hierarchy = run_leiden(element_summaries)
        # The community_map_to_cluster is a dictionary of node to cluster mapping. The hierarchy is a dictionary of cluster to parent cluster mapping.
        
        community_graph = nx.DiGraph()

        # Add nodes with cluster info
        for level, communities in community_map_to_cluster.items():
            for cluster_id, node_list in communities.items():
                for node_id in node_list:
                    # Find the node_id in the element_summaries graph and add it to the community_graph
                    
                    # Convert the node_id to an all uppercase string if it's a string. If it's an int, convert it to a string.
                    # converted_node_id = node_id.upper() if isinstance(node_id, str) else str(node_id)
                    if node_id in element_summaries.nodes:
                        community_graph.add_node(node_id, level=level, cluster=cluster_id)
                    else:
                        logger.warning(f"Node {node_id} not found in element_summaries graph")

        # Add edges indicating parent-child relationships
        for cluster_id, parent_cluster_id in hierarchy.items():
            if parent_cluster_id != -1:
                community_graph.add_edge(parent_cluster_id, cluster_id)

        return community_graph

    def community_summaries(self, graph_communities: nx.DiGraph, element_summaries: nx.Graph):
        # Create summaries for each community
        summary_graph_file_path = "graphs/community_summaries.gpickle"
        try:
            if os.path.exists(summary_graph_file_path):
                pickled_graph = pickle.load(open(summary_graph_file_path, "rb"))
                return pickled_graph
        except:
            logger.info("No pickled graph found. Generating a new graph.")
            pass
        
        summary_graph = nx.DiGraph()
        cluster_to_nodes: Dict[int, List[nx.classes.reportviews.NodeView]] = collections.defaultdict(list)

        # Collect nodes by cluster
        for node, data in graph_communities.nodes(data=True):
            cluster_id = data.get("cluster")
            if cluster_id is not None:
                cluster_to_nodes[cluster_id].append(node)

        # Create a summary for each cluster
        for cluster_id, node_list in cluster_to_nodes.items():
            relevant_nodes: List[nx.classes.reportviews.NodeView] = list()
            for n in node_list:
                if n in element_summaries.nodes:
                    relevant_nodes.append(element_summaries.nodes[n])
                    
            chat_session = self.community_summarizer_model.start_chat()
            formatted_message = COMMUNITY_SUMMARIZATION_PROMPT.format(
                community_nodes=relevant_nodes
            )
            response = chat_session.send_message(formatted_message)
            summary = response.text if response.text else "No summary provided"
            summary_graph.add_node(cluster_id, summary=summary)

        # Replicate edges indicating parent-child relationships between clusters
        for parent_cluster_id, child_cluster_id in graph_communities.edges():
            summary_graph.add_edge(parent_cluster_id, child_cluster_id)
            
        # pickle.dump(summary_graph, open(summary_graph_file_path, "wb"))

        return summary_graph

if __name__ == "__main__":
    # Example usage of the GraphRAG class
    graph_rag = GraphRAG()
    documents = graph_rag.read_documents("data/")
    text_chunks = graph_rag.text_chunks(documents)
    element_instances = graph_rag.element_instances(text_chunks)
    
    element_summaries = graph_rag.element_summaries(element_instances)
    
    graph_communities = graph_rag.graph_communities(element_instances)
    community_summaries = graph_rag.community_summaries(graph_communities, element_summaries)

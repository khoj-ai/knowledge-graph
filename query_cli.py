"""
Interactive CLI for querying all ChromaDB collections with conversation context
"""

import chromadb
import json
import networkx as nx
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from typing import Dict, List
import typer
from typing_extensions import Annotated
from graph import GraphRAG
from prompts_templates import FINAL_RESPONSE_COMPOSER_PROMPT, FINAL_RESPONSE_COMPOSER_SYSTEM_PROMPT, QUERY_COMPOSER_SYSTEM_PROMPT, GRAPH_QUERY_TRANSFORM_PROMPT
import google.generativeai as genai # type: ignore

from google.generativeai.types.safety_types import ( # type: ignore
    HarmBlockThreshold,
    HarmCategory,
)

from utils import clean_json

console = Console()
app = typer.Typer()

MAX_DOCUMENT_LENGTH = 1000

class LLMSearch:
    def __init__(self):
        self.query_composer = genai.GenerativeModel(
            "gemini-1.5-flash-002",
            system_instruction=QUERY_COMPOSER_SYSTEM_PROMPT,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
    def transform_query(self, query: str, nodes: nx.classes.reportviews.NodeView) -> List[str]:
        """Transform the query into a search query"""
        node_packets = []
        for node in nodes:
            node_packets.append({
                "name": node[0],
                "type": node[1].get('type', 'unknown'),
                "description": node[1].get('description', '')
            })
        
        query_data = {
            "entities": ", ".join([f"{node['name']}: ({node['description']})" for node in node_packets]),
            "input_text": query
        }
        
        chat_session = self.query_composer.start_chat()
        formatted_message = GRAPH_QUERY_TRANSFORM_PROMPT.format(**query_data)
        
        response = chat_session.send_message(formatted_message)
        
        try:
            cleaned_response = clean_json(response.text)
            inferred_queries = json.loads(cleaned_response)
            return inferred_queries['queries']
        except Exception as e:
            return [query]
        
class LLMResponse:
    def __init__(self):
        self.respond = genai.GenerativeModel(
            "gemini-1.5-flash-002",
            system_instruction=FINAL_RESPONSE_COMPOSER_SYSTEM_PROMPT,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
    def respond_to_query(self, queries: List[str], results) -> str:
        relevant_docs: Dict[str, List[str]] = {}
        for result in results:
            for id, document in zip(result['ids'][0], result['documents'][0]):
                if str(id) not in relevant_docs:
                    relevant_docs[str(id)] = []
                relevant_docs[str(id)].append(document[:MAX_DOCUMENT_LENGTH])
            
        query_data = {
            "input_text": queries,
            "relevant_docs": relevant_docs
        }
        
        chat_session = self.respond.start_chat()
        formatted_message = FINAL_RESPONSE_COMPOSER_PROMPT.format(**query_data)
        
        response = chat_session.send_message(formatted_message)
        
        return response.text
        

def get_chroma_client():
    """Get ChromaDB client connection"""
    return chromadb.HttpClient(host='localhost', port=8000)

def format_collection_results(collection_name: str, results, query: str) -> Table:
    """Format search results for a single collection"""
    table = Table(title=f"Results from {collection_name}")
    
    table.add_column("ID", style="cyan")
    table.add_column("Content", style="green")
    table.add_column("Distance", style="magenta")

    for id, document, distance in zip(
        results['ids'][0], 
        results['documents'][0], 
        results['distances'][0]
    ):
        table.add_row(
            str(id),
            document if len(document) < MAX_DOCUMENT_LENGTH else f"{document[:MAX_DOCUMENT_LENGTH]}...",
            f"{distance:.4f}"
        )
    
    return table

def search_collections(query: str, n_results: int = 3) -> List[chromadb.QueryResult]:
    """Search across all collections"""
    
    aggregated_results = []
    
    try:
        client = get_chroma_client()
        collections = client.list_collections()
        
        results_found = False
        
        console.print(f"[bold blue]Querying for '{query}'[/bold blue]")
        
        for collection in collections:
            coll = client.get_collection(collection)
            results = coll.query(
                query_texts=[query],
                n_results=n_results
            )
            
            aggregated_results.append(results)
            
            if results['ids'][0]:
                results_found = True
                table = format_collection_results(collection, results, query)
                # console.print(table)
                console.print()
        
        if not results_found:
            console.print("[yellow]No results found in any collection[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        
    return aggregated_results

@app.command()
def interactive(
    n_results: Annotated[int, typer.Option("--n-results", "-n")] = 7,
    transform_queries: bool = typer.Option(True, "--transform-queries", "-transform", help="Use LLM to transform queries"),
    use_llm: bool = typer.Option(True, "--use-llm", "-llm", help="Use LLM to respond to query with collected data")
):
    """Start interactive query session"""
    console.print("[bold green]Welcome to GraphRAG Query CLI![/bold green]")
    console.print("Enter your queries below. Type 'exit' to quit.\n")
    
    graph_rag = GraphRAG()
    
    while True:
        query = Prompt.ask("[bold blue]Query[/bold blue]")
        queries = [query]
        
        if query.lower() in ('exit', 'quit'):
            break
            
        if transform_queries:
            documents = graph_rag.read_documents("data/")
            text_chunks = graph_rag.text_chunks(documents)
            element_instances = graph_rag.element_instances(text_chunks)
            
            nodes = element_instances.nodes(data=True)
            llm = LLMSearch()
            inferred_queries = llm.transform_query(query, nodes)
            queries.extend(inferred_queries)
            
            for inferred_query in inferred_queries:
                aggregated_results = search_collections(inferred_query, n_results)
                console.print()
        else:
            aggregated_results = search_collections(query, n_results)
            
        
        if use_llm:
            llm_response = LLMResponse()
            response = llm_response.respond_to_query(queries, aggregated_results)
            console.print("[bold green]Response[/bold green]")
            console.print(response)
            
        console.print()

if __name__ == "__main__":
    app()

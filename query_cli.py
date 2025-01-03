"""
Interactive CLI for querying all ChromaDB collections with conversation context
"""

import chromadb
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from typing import Dict, List
import typer
from typing_extensions import Annotated

console = Console()
app = typer.Typer()

MAX_DOCUMENT_LENGTH = 1000

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

def search_collections(query: str, n_results: int = 3):
    """Search across all collections"""
    try:
        client = get_chroma_client()
        collections = client.list_collections()
        
        results_found = False
        
        for collection in collections:
            coll = client.get_collection(collection)
            results = coll.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if results['ids'][0]:
                results_found = True
                table = format_collection_results(collection, results, query)
                console.print(table)
                console.print()
        
        if not results_found:
            console.print("[yellow]No results found in any collection[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@app.command()
def interactive(
    n_results: Annotated[int, typer.Option("--n-results", "-n")] = 3
):
    """Start interactive query session"""
    console.print("[bold green]Welcome to GraphRAG Query CLI![/bold green]")
    console.print("Enter your queries below. Type 'exit' to quit.\n")
    
    while True:
        query = Prompt.ask("[bold blue]Query[/bold blue]")
        
        if query.lower() in ('exit', 'quit'):
            break
            
        search_collections(query, n_results)
        console.print()

if __name__ == "__main__":
    app()

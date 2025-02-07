# Graph RAG

[GraphRAG](https://microsoft.github.io/graphrag/) is a technique proposed by Microsoft Research to generate knowledge graphs from text using LLMs. This project is an implementation of GraphRAG in Python. It makes it simple to get up and running with GraphRAG and generate knowledge graphs from text. It also provides a semantic search interface to query the knowledge graph.

## Install
This project uses [uv](https://docs.astral.sh/uv/getting-started/) for Python dependency management. 

To install uv, follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

Install the dependencies with the following command:

```bash
uv run pip install
```

## Usage

1. Add the files you want to add to your knowledge graph in the `data` folder.

2. Set your `GOOGLE_API_KEY` in the `.env` file. You can get it from [here](https://aistudio.google.com/app/apikey).

3. Run the following command to generate the knowledge graph:

First, start the Chroma server:
```bash
chroma run --path db
```

```bash
python3 graph.py
```

4. The knowledge graph will be saved in a chromadb database in the `db` folder. To query it, run

```bash
python3 query_cli.py
```

5. To visualize the knowledge graph, run

```bash
streamlit run visualize.py
```
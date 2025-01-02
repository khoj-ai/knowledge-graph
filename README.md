# Graph RAG

## Install
This project uses [uv](https://docs.astral.sh/uv/getting-started/) for Python dependency management. 

To install uv, follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

Install the dependencies with the following command:

```bash
uv run pip install
```

## Usage

1. Add the files you want to add to your knowledge graph in the `data` folder.

2. Run the following command to generate the knowledge graph:

```bash
python3 graph.py
```

3. The knowledge graph will be saved in a chromadb database in the `db` folder. To query it, run

```bash
python3 query_cli.py
```
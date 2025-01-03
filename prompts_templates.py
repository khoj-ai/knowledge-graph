GRAPH_EXTRACTION_SYSTEM_PROMPT = """
You are a trained export in extracting entities and relationships from text documents. Your goal is to identify all entities and relationships from the text provided.
"""

GRAPH_EXTRACTION_JSON_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity output as a JSON entry with the following format:

{{"name": <entity name>, "type": <type>, "description": <entity description>}}

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: an integer score between 1 to 10, indicating strength of the relationship between the source entity and target entity
Format each relationship as a JSON entry with the following format:

{{"source": <source_entity>, "target": <target_entity>, "relationship": <relationship_description>, "relationship_strength": <relationship_strength>}}

3. Return output in {language} as a single list of all JSON entities and relationships identified in steps 1 and 2.

-Real Data-
######################
entity_types: {entity_types}
text: {input_text}
######################
output:"""

NODE_SUMMARIZATION_SYSTEM_PROMPT = """
You are a data scientist working on a project that involves analyzing a graph data structure. Your goal is to summarize the nodes in the graph based on their attributes and relationships.
"""

NODE_SUMMARIZATION_PROMPT = """
-Goal-
You are give information of a node, its name, a description, and the relationships it has with other nodes. Your task is to summarize the node based on this information in a few sentences or bullet points.

--Real Data--
######################
node_name: {node_name}
node_type: {node_type}
node_description: {node_description}
relationships: {relationships}
######################
output:
"""


COMMUNITY_SUMMARIZATION_SYSTEM_PROMPT = """
You are a data scientist working on a project that involves analyzing a graph data structure. Your goal is to summarize the communities in the graph based on their attributes and relationships.
"""

COMMUNITY_SUMMARIZATION_PROMPT = """
-Goal-
You are given a collection of nodes that are part of a community in a graph. Your task is to summarize the community based on the nodes' attributes and descriptions. Be detailed and process in your summary. Include any patterns, commonalities, or relationships among the nodes in the community. You will receive a list of descriptions for each node in the community. You must rely only on this information to summarize the community.

--Real Data--
######################
community_nodes: {community_nodes}
######################
output:
"""

QUERY_COMPOSER_SYSTEM_PROMPT = """
You are a data scientist working on a project that involves querying a graph data structure. Your goal is to compose queries that will retrieve specific information from the graph.
"""

GRAPH_QUERY_TRANSFORM_PROMPT = """
-Goal-
Given a query and a list of entity types, transform the query into a search query that is specific to the entities in the graph.

-Steps-

1. Examine all the entity options and the query provided. Decide which of the entities is most relevant to answering the question, and construct a search query that will retrieve information about that entity in order to answer the input query. Queries should be questions or statements that can be used to retrieve information about the entities in the graph.
2. Return output in English as a single list of all search queries gathered in step 1 that can assist in answering the input query. It should be in the following format:

{{"queries": [<search_query1>, <search_query2>, ...]}}

--Real Data--
######################
entities: {entities}
input_query: {input_text}
######################
output:
"""

FINAL_RESPONSE_COMPOSER_SYSTEM_PROMPT = """
You are an astute analyst who is tasked with extracting relevant information from a set of documents, given a query. Your goal is to generate a response that answers the query using the information extracted from the documents.
"""

FINAL_RESPONSE_COMPOSER_PROMPT = """
-Goal-
Given a set of queries and a list of relevant documents, generate a response that answers the query using the information extracted from the documents. You can include citations or references to the relevant documents in your response, denoted by the document ID.

-Steps-

1. Analyze the queries and the relevant documents provided. Identify the key information in the documents that can help answer the query.
2. Compose a response that directly addresses the query using the information extracted from the documents. You can paraphrase or summarize the information, but ensure that the response is clear and relevant to the query.

--Real Data--
######################
input_query: {input_text}
relevant_docs: {relevant_docs}
######################
output:
"""
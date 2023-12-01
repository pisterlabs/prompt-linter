"""
graph_handling.py

Contains functions & classes for creating graphs and pulling information from them.
"""

from typing import List, Optional

from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.graphs import Neo4jGraph
from langchain.graphs.graph_document import (
    GraphDocument,
    Node as BaseNode,
    Relationship as BaseRelationship,
)
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import Document
from tqdm import tqdm

import embedding_handling
import file_handling
from load_environment import load_environment


load_environment()


GPT3_TURBO = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
GPT4_TURBO = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)

GPT3 = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
GPT4 = ChatOpenAI(model="gpt-4", temperature=0)

NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "athena_password"


class Property(BaseModel):
    """A single property consisting of key and value"""

    key: str = Field(..., description="key")
    value: str = Field(..., description="value")


class Node(BaseNode):
    properties: Optional[List[Property]] = Field(
        None, description="List of node properties"
    )


class Relationship(BaseRelationship):
    properties: Optional[List[Property]] = Field(
        None, description="List of relationship properties"
    )


class KnowledgeGraph(BaseModel):
    """Generate a knowledge graph with entities and relationships."""

    nodes: List[Node] = Field(..., description="List of nodes in the knowledge graph")
    rels: List[Relationship] = Field(
        ..., description="List of relationships in the knowledge graph"
    )


class NodeNameList(BaseModel):
    """A list of the names of knowledge graph nodes."""

    names: list[str] = Field(
        ..., description="List of desired node names from a knowledge graph"
    )


def format_property_key(string: str) -> str:
    """Format property keys into snake case."""

    words = [word.lower() for word in string.split()]

    if not words:
        return string.lower()

    return "_".join(words)


def props_to_dict(props: list[Property]) -> dict:
    """Convert properties to a dictionary."""

    properties = {}

    if not props:
        return properties

    for prop in props:
        properties[format_property_key(prop.key)] = prop.value

    return properties


def map_to_base_node(node: Node) -> BaseNode:
    """Map the KnowledgeGraph Node to the base Node."""

    properties = props_to_dict(node.properties) if node.properties else {}
    # Add name property for better Cypher statement generation
    properties["name"] = node.id.title()

    return BaseNode(
        id=node.id.title(), type=node.type.capitalize(), properties=properties
    )


def map_to_base_relationship(rel: Relationship) -> BaseRelationship:
    """Map the KnowledgeGraph Relationship to the base Relationship."""

    source = map_to_base_node(rel.source)
    target = map_to_base_node(rel.target)
    properties = props_to_dict(rel.properties) if rel.properties else {}

    return BaseRelationship(
        source=source, target=target, type=rel.type, properties=properties
    )


def get_extraction_chain(
    llm: ChatOpenAI,
    allowed_nodes: Optional[List[str]] = None,
    allowed_rels: Optional[List[str]] = None,
    verbose: bool = False,
):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""# Knowledge Graph Instructions for GPT
## 1. Overview
You are a top-tier algorithm designed for extracting information from markdown notes in structured formats to build a knowledge graph.
- **Nodes** represent entities and concepts. They're akin to Wikipedia nodes.
- The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.
## 2. Labeling Nodes
- **Consistency**: Ensure you use basic or elementary types for node labels.
  - For example, when you identify an entity representing a person, always label it as **"person"**. Avoid using more specific terms like "mathematician" or "scientist".
- **Node IDs**: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers found in the text.
{'- **Allowed Node Labels:**' + ", ".join(allowed_nodes) if allowed_nodes else ""}
{'- **Allowed Relationship Types**:' + ", ".join(allowed_rels) if allowed_rels else ""}
## 3. Handling Numerical Data and Dates
- Numerical data, like age or other related information, should be incorporated as attributes or properties of the respective nodes.
- **No Separate Nodes for Dates/Numbers**: Do not create separate nodes for dates or numerical values. Always attach them as attributes or properties of nodes.
- **Property Format**: Properties must be in a key-value format.
- **Quotation Marks**: Never use escaped single or double quotes within property values.
- **Naming Convention**: Use camelCase for property keys, e.g., `birthDate`.
## 4. Coreference Resolution
- **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.
If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"),
always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.
Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.
## 5. Strict Compliance
Adhere to the rules strictly. Non-compliance will result in termination.
          """,
            ),
            (
                "human",
                "Use the given format to extract information from the following input: {input}",
            ),
            ("human", "Tip: Make sure to answer in the correct format"),
        ]
    )

    return create_structured_output_chain(KnowledgeGraph, llm, prompt, verbose=verbose)


def get_graph_connector() -> Neo4jGraph:
    """Returns a wrapper for the Neo4j database."""

    return Neo4jGraph(
        url=NEO4J_URL,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
    )


def delete_graph(are_you_sure: bool) -> None:
    """This will wipe all nodes & relationships from the Neo4j Graph."""

    if are_you_sure:
        result = get_graph_connector().query("MATCH (n) DETACH DELETE n")

        if len(result) == 0:
            print("Neo4j database emptied.")
        else:
            print("Delete query returned results. Something may have gone wrong.")


def get_knowledge_graph_from_chunk(
    chunk: Document,
    llm: ChatOpenAI,
    allowed_nodes: list[str] | None = None,
    allowed_rels: list[str] | None = None,
    verbose: bool = False,
) -> KnowledgeGraph:
    """Runs the LLM function to extract a Knowledge Graph from a document chunk."""

    return get_extraction_chain(llm, allowed_nodes, allowed_rels, verbose).run(
        chunk.page_content
    )


def create_graph_document_from_note(
    file_name: str,
    llm: ChatOpenAI,
    allowed_nodes: list[str] | None = None,
    allowed_rels: list[str] | None = None,
    verbose: bool = False,
) -> GraphDocument:
    file_store = file_handling.load_file_store()

    if file_store is None:
        print("Failed to retrieve file store. Exiting graph creation.")
        return

    collection = embedding_handling.get_vector_store_collection()

    doc, chunks = embedding_handling.get_chunks_from_file_name(file_name)

    # make vault node
    vault_node = BaseNode(id="ObsidianVault", type="ObsidianVaultNode")

    # make note node
    note_node = BaseNode(
        id=file_store[file_name]["uuid"],
        type="ObsidianNote",
        properties={"file_name": file_name},
    )

    # vault to note relationship
    vault_note_relationship = BaseRelationship(
        source=vault_node, target=note_node, type="contains_note"
    )

    all_base_nodes = [vault_node, note_node]
    all_base_relationships = [vault_note_relationship]

    # get knowledge graph from chunks
    for i, chunk in tqdm(
        enumerate(chunks),
        desc="Creating graph from each document chunk",
        total=len(chunks),
    ):
        chunk_kg = get_knowledge_graph_from_chunk(
            chunk, llm, allowed_nodes, allowed_rels, verbose
        )

        # convert knowledge graph into base nodes & base relationships
        base_nodes = [map_to_base_node(node) for node in chunk_kg.nodes]
        base_relationships = [map_to_base_relationship(rel) for rel in chunk_kg.rels]

        # make chunk node
        chunk_node = BaseNode(
            id=file_store[file_name]["chunks"][i],
            type="ObsidianNoteChunk",
            properties={
                "file_name": file_name,
                "chunk_number": i,
                "embeddings": collection.get(
                    ids=file_store[file_name]["chunks"][i], include=["embeddings"]
                )["embeddings"][0],
            },
        )

        # add relationship between note node and chunk node
        note_to_chunk_relationship = BaseRelationship(
            source=note_node, target=chunk_node, type="contains_chunk"
        )

        # add relationships between chunk nodes and GPT-generated nodes
        chunk_to_node_relationships = []
        for node in base_nodes:
            chunk_to_node_relationships.append(
                BaseRelationship(source=chunk_node, target=node, type="references_node")
            )

        # collect all nodes & relationships
        all_base_nodes += base_nodes + [chunk_node]
        all_base_relationships += (
            base_relationships
            + chunk_to_node_relationships
            + [note_to_chunk_relationship]
        )

    # assemble nodes & relationships into GraphDocument
    graph_document = GraphDocument(
        nodes=all_base_nodes, relationships=all_base_relationships, source=doc
    )

    return graph_document
    # later, graph.add_graph_documents([graph_document])


def get_all_node_names() -> list[str]:
    """Returns a list of all the names of the nodes in the graph"""

    names: list[dict] = get_graph_connector().query(
        """
        MATCH (n)
        WHERE n.name IS NOT NULL
        RETURN n.name"""
    )

    return [list(d.values())[0] for d in names]


def get_relevant_nodes_from_question(
    llm: ChatOpenAI,
    node_name_list: list[str],
    question: str,
    verbose: bool = False,
) -> NodeNameList:
    """
    Uses LLM to shorten & sort a list of node names by how relevant they are to a
    user question. This *does* appear to use the LLM's previous knowledge.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """# Prompt for GPT-4:
Question:
"{question}"

List of Node Names from the Knowledge Graph:
{names}

# Task for GPT-4:
Analyze the provided list of names from the knowledge graph in the context of the question. Identify and list the names that are most relevant to the question, ordering them from the most important to less important. Do not include names that are not very important. Consider only the content of the question and do not use prior knowledge.
""",
            ),
            ("human", "Tip: Make sure to answer in the correct format"),
        ]
    )

    chain = create_structured_output_chain(NodeNameList, llm, prompt, verbose=verbose)

    return chain.run(question=question, names=", ".join(node_name_list))


def get_chunk_ids_by_node_names(node_names: list[str]) -> list[str]:
    """Given a list of node names, returns the ids of the chunks that reference them.

    May contain duplicates.
    """

    if len(node_names) == 0:
        return []

    # This query will collect ids even if there are duplicates of the named node
    ids: list[dict] = get_graph_connector().query(
        f"""
                            MATCH (n)
                            WHERE n.name IN [{",".join([f'"{name}"' for name in node_names])}]
                            OPTIONAL MATCH (n)-[r]-(related:ObsidianNoteChunk)"""
        + """ RETURN collect({id: related.id}) as relatedNodes
                        """
    )

    return [list(d.values())[0] for d in ids[0]["relatedNodes"]]


def get_non_housekeeping_relationships_from_node_name(
    node_name: str,
    allowed_names: list[str] | None = None,
) -> list[tuple[dict, str, dict]]:
    """
    Given a node name, will return the relationships between that node and the other
    non-housekeeping nodes in the graph.

    If a list of names is provided in allowed_names, will only return the relationships
    that occur between the primary node and any of the allowed nodes.
    """

    allowed_node_syntax = ""

    # Block of Cypher for modifying the results to blank out non-allowed nodes
    if allowed_names:
        allowed_node_syntax += "WHEN NOT related.name IN ["
        allowed_node_syntax += ",".join([f'"{name}"' for name in allowed_names])
        allowed_node_syntax += "]\nTHEN {label: 'Unrelated'}"

    query_results: list[dict] = get_graph_connector().query(
        f"""
                MATCH (n)
                WHERE n.name = '{node_name}'"""
        + """
                OPTIONAL MATCH (n)-[r]-(related)
                RETURN n,
                    collect(r) as relationships,
                    collect(
                        CASE
                            WHEN 'ObsidianNoteChunk' IN labels(related)
                                THEN {label: 'ObsidianNoteChunk'}"""
        + allowed_node_syntax
        + """
                            ELSE related
                        END
                    ) as relatedNodes
                """
    )

    results: list[tuple[dict, str, dict]] = []

    # could be more than one node with the same name
    for node in query_results:
        for relationship in node["relationships"]:
            # second item is edge type
            # len checks are to make sure the node didn't get filtered out
            if (
                relationship[1] != "REFERENCES_NODE"
                and len(relationship[0]) != 0
                and len(relationship[2]) != 0
            ):
                results.append(relationship)

    return results


def get_interrelationships_between_nodes(
    node_names: list[str],
) -> list[tuple[dict, str, dict]]:
    """Given a list of node names, will return the relationships between them."""

    node_str = ",".join([f'"{node}"' for node in node_names])

    query_results: list[dict] = get_graph_connector().query(
        f"""
        UNWIND [{node_str}] AS nodeName1
        UNWIND [{node_str}] AS nodeName2
        MATCH (n1)-[r]->(n2)
        WHERE n1.name = nodeName1 AND n2.name = nodeName2
        RETURN n1, r, n2
        """
    )

    results: list[tuple[dict, str, dict]] = []

    # one row per relationship
    for row in query_results:
        results.append(row["r"])

    return results


def summarize_relationship(
    llm: ChatOpenAI,
    relationship: tuple[dict, str, dict],
    verbose: bool = False,
) -> str:
    """Uses LLM to summarize the relationship between two nodes."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """# Prompt for GPT-4:
Relationship between two nodes:
"{relationship}"

# Task for GPT-4:
This is a relationship between two nodes in a Neo4j graph. Please use this information to give a summary of this relationship in a succinct paragraph that does not mention anything about a graph or nodes.
""",
            ),
        ]
    )

    chain = create_structured_output_chain(str, llm, prompt, verbose=verbose)

    return chain.run(relationship=relationship)

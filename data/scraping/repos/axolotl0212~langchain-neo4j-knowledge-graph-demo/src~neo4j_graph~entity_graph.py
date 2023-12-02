from typing import List, Tuple, Literal
from neo4j import (
    GraphDatabase,
    Driver,
)
import networkx as nx
from langchain.graphs.networkx_graph import (
    NetworkxEntityGraph,
    KnowledgeTriple,
)
from config import (
    NEO4J_DB_URI,
    NEO4J_DB_NAME,
    NEO4J_DB_USER,
    NEO4J_DB_PASS,
)
from .repository import (
    add_edge_and_node_if_not_exists,
    clear_all,
    delete_edge,
    find_all_edges,
    find_surrounding_edges,
    fulltext_search_surrounding_edges,
)
from .model import (
    Node,
    Edge,
)
from .utils import add_neo4j_edges_to_networkx_graph

SearchMode = Literal["exact", "fulltext"]


class Neo4jEntityGraph(NetworkxEntityGraph):
    driver: Driver
    search_mode: SearchMode

    def __init__(
        self,
        *,
        search_mode: SearchMode = "fulltext",
        verbose: bool = True,
    ):
        print("Initiating neo4j driver...")
        self.driver = GraphDatabase.driver(
            uri=NEO4J_DB_URI,
            auth=(NEO4J_DB_USER, NEO4J_DB_PASS),
            database=NEO4J_DB_NAME,
        )
        self.search_mode = search_mode
        print("Initiated entity graph.")

    def __del__(self):
        print("Closing neo4j driver...")
        self.driver.close()

    def add_triple(self, knowledge_triple: KnowledgeTriple) -> None:
        print(f"Adding triple {knowledge_triple}...")
        with self.driver.session() as session:
            session.execute_write(
                add_edge_and_node_if_not_exists,
                src=Node(knowledge_triple.subject),
                relation=Edge(knowledge_triple.predicate),
                sink=Node(knowledge_triple.object_),
            )

    def delete_triple(self, knowledge_triple: KnowledgeTriple) -> None:
        print(f"Deleting triple {knowledge_triple}...")
        with self.driver.session() as session:
            session.execute_write(
                delete_edge,
                src=Node(knowledge_triple.subject),
                relation=Edge(knowledge_triple.predicate),
                sink=Node(knowledge_triple.object_),
            )

    def get_triples(self) -> List[Tuple[str, str, str]]:
        with self.driver.session() as session:
            resp = session.execute_read(find_all_edges)
            triples: List[Tuple[str, str, str]] = []
            for record in resp:
                triples.append(
                    (
                        record.src["name"],
                        record.sink["name"],
                        record.relation["name"],
                    )
                )
            return triples

    def get_entity_knowledge(self, entity: str, depth: int = 1) -> List[str]:
        print(f"Searching knowledge graph for entity {entity}")
        with self.driver.session() as session:
            if self.search_mode == "exact":
                search_func = find_surrounding_edges
            elif self.search_mode == "fulltext":
                search_func = fulltext_search_surrounding_edges
            else:
                raise NotImplementedError(
                    f"search mode {self.search_mode} is not supported"
                )
            resp = session.execute_read(
                search_func,
                q=entity,
                depth=depth,
            )

        print("Found the following from knowledge graph:")
        statements: List[str] = []
        for record in resp:
            statement = record.to_statement()
            print(statement)
            statements.append(statement)

        return statements

    def write_to_gml(self, path: str) -> None:
        print("Loading the entire knowledge graph...")
        with self.driver.session() as session:
            resp = session.execute_read(
                find_all_edges,
            )

        print("Constructing NetworkX graph...")
        networkx_graph = nx.DiGraph()
        add_neo4j_edges_to_networkx_graph(
            graph=networkx_graph,
            edge_records=resp,
        )

        print(f"Writing gml file to path: {path}...")
        nx.gml.write_gml(networkx_graph, path=path)
        print("Written graph to gml file.")

    def clear(self) -> None:
        print("Clearing the entire knowledge graph...")
        with self.driver.session() as session:
            session.execute_write(clear_all)

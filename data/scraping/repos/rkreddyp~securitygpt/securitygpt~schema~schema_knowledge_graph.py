### StockJsons
from enum import Enum, IntEnum
from pydantic import Field
from typing import List, Dict, Optional, Type, Union
from securitygpt.schema.schema_base_openai import OpenAISchema
from pydantic import BaseModel


class Node(BaseModel):
    """
    Node class for the knowledge graph. Each node represents an entity.
    
    Attributes:
        id (int): Unique identifier for the node.
        label (str): Label or name of the node.
        color (str): Color of the node.
        num_targets (int): Number of target nodes connected to this node.
        num_sources (int): Number of source nodes this node is connected from.
        list_target_ids (List[int]): List of unique identifiers of target nodes connected to this node.
    """
    
    id: int
    label: str
    color: str
    num_targets: int
    num_sources: int
    list_target_ids: List[int] = Field(default_factory=list)

class Edge(BaseModel):
    source: int
    target: int
    label: str
    color: str = "black"

class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
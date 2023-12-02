from dataclasses import dataclass
from enum import Enum

from ...core.contracts import OpenAIApiType


class MLIndexKind(str, Enum):
    FAISS = 'faiss'
    ACS = 'acs'


class MLIndexEmbeddingKind(str, Enum):
    OPENAI = 'open_ai'


class MLIndexConnectionType(str, Enum):
    WORKSPACE_KEYVAULT = 'workspace_keyvault'
    WORKSPACE_CONNECTION = 'workspace_connection'
    ENVIRONMENT = 'environment'


@dataclass
class FieldMapping:
    content: str = None
    embedding: str = None
    filename: str = None
    metadata: str = None
    title: str = None
    url: str = None


@dataclass
class MLIndexConnection:
    id: str = None
    key: str = None
    resource_group: str = None
    subscription: str = None
    workspace: str = None


@dataclass
class MLIndexSectionBase:
    endpoint: str = None
    api_base: str = None
    api_version: str = None
    connection: MLIndexConnection = None
    connection_type: MLIndexConnectionType = None


@dataclass
class IndexSection(MLIndexSectionBase):
    kind: MLIndexKind = None
    engine: str = None
    method: str = None
    index: str = None
    field_mapping: FieldMapping = None


@dataclass
class EmbeddingsSection(MLIndexSectionBase):
    kind: MLIndexEmbeddingKind = None
    model: str = None
    deployment: str = None
    api_type: OpenAIApiType = None
    batch_size: int = None
    dimension: int = None
    schema_version: str = None


@dataclass
class MLIndexYamlConfig:
    index: IndexSection = None
    embeddings: EmbeddingsSection = None

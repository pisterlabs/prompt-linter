from typing import Any, Union

from promptflow.connections import AzureOpenAIConnection, OpenAIConnection, CognitiveSearchConnection
from .connection_handler import ConnectionHandler
from ...utils.workspace_connection_manager import WorkspaceConnectionManager
from ...contracts.ml_index_yaml_config import MLIndexConnection


class WorkspaceConnectionHandler(ConnectionHandler):

    @staticmethod
    def get_key(connection: MLIndexConnection) -> Any:
        ws_connection_manager: WorkspaceConnectionManager = WorkspaceConnectionManager()
        ws_connection: Union[AzureOpenAIConnection, OpenAIConnection, CognitiveSearchConnection]
        ws_connection = ws_connection_manager.get_connection_with_id(connection.id)
        return ws_connection.api_key

    @staticmethod
    def get_api_base(connection: MLIndexConnection) -> Any:
        ws_connection_manager: WorkspaceConnectionManager = WorkspaceConnectionManager()
        ws_connection: Union[AzureOpenAIConnection, OpenAIConnection, CognitiveSearchConnection]
        ws_connection = ws_connection_manager.get_connection_with_id(connection.id)
        if isinstance(ws_connection, AzureOpenAIConnection) or isinstance(ws_connection, CognitiveSearchConnection):
            return ws_connection.api_base
        else:
            return None

    @staticmethod
    def get_api_type(connection: MLIndexConnection) -> str:
        ws_connection_manager: WorkspaceConnectionManager = WorkspaceConnectionManager()
        ws_connection: Union[AzureOpenAIConnection, OpenAIConnection, CognitiveSearchConnection]
        ws_connection = ws_connection_manager.get_connection_with_id(connection.id)

        from ....core.contracts.types import OpenAIApiType
        if isinstance(ws_connection, AzureOpenAIConnection):
            return OpenAIApiType.AZURE.value
        elif isinstance(ws_connection, OpenAIConnection):
            return OpenAIApiType.OPENAI.value
        else:
            return None

    @staticmethod
    def get_api_version(connection: MLIndexConnection) -> Any:
        ws_connection_manager: WorkspaceConnectionManager = WorkspaceConnectionManager()
        ws_connection: Union[AzureOpenAIConnection, OpenAIConnection, CognitiveSearchConnection]
        ws_connection = ws_connection_manager.get_connection_with_id(connection.id)
        if isinstance(ws_connection, AzureOpenAIConnection) or isinstance(ws_connection, CognitiveSearchConnection):
            return ws_connection.api_version
        else:
            return None

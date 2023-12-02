from typing import List

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.callbacks.base import BaseCallbackManager
from langchain.tools import BaseTool

from tools.k8s_sme.tools import KubernetesSMEModel, KubernetesSMETool

class KubernetesSMEToolkit(BaseToolkit):
    """Toolkit for interacting with git repositories."""

    model: KubernetesSMEModel
    callback_manager: BaseCallbackManager | None

    class Config:
        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Return a list of tools."""
        return [
            KubernetesSMETool(model=self.model, callback_manager=self.callback_manager),
        ]
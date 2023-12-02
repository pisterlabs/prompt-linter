from typing import List
from tools.code_generator.tools import CodeGeneratorModel, DesignPatternGetTemplateTool, DesignPatternListTemplatesTool, DesignPatternListTool

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.callbacks.base import BaseCallbackManager
from langchain.tools import BaseTool

class CoderToolkit(BaseToolkit):
    """Toolkit for performing coding tasks related to kubernetes."""
    model: CodeGeneratorModel
    callback_manager: BaseCallbackManager | None

    class Config:
        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Return a list of tools."""
        return [
            DesignPatternListTool(model=self.model, callback_manager=self.callback_manager),
            DesignPatternListTemplatesTool(model=self.model, callback_manager=self.callback_manager),
            DesignPatternGetTemplateTool(model=self.model, callback_manager=self.callback_manager),
        ]
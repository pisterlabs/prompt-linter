"""Diet Plan Toolkit Object

Toolkit object for diet plan related tools

"""
from typing import List, Optional
from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool

from healthfirstai_prototype.models.data_models import User
from .tools import (
    DietPlanTool,
    EditDietPlanTool,
    EditBreakfastTool,
    EditLunchTool,
    EditDinnerTool,
    BreakfastTool,
    LunchTool,
    DinnerTool,
)


# NOTE: I'm not sure if there is a need to use BaseToolkit
# I think we need to learn more about dataclasses to decide
class DietPlanToolkit(BaseToolkit):
    """Diet Plan Toolkit

    Parameters:
        user_info (User): The user's information.

    Methods:
        get_tools: Get the tools in the toolkit.
    """

    user_info: Optional[User] = None

    # NOTE: In future, by using this DietPlanToolkit class,
    # We can pass certain user params useful for diet plan
    # related tasks
    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            DietPlanTool(),
            EditDietPlanTool(),
            EditBreakfastTool(),
            EditLunchTool(),
            EditDinnerTool(),
            BreakfastTool(),
            LunchTool(),
            DinnerTool(),
        ]

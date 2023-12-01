from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from ..utils import getOcean


class ToolInputSchema(BaseModel):
    placeholder: str = Field(..., description="Just fill in `asdf`")


def get_reward_distribution(placeholder: str) -> str:
    try:
        return getOcean().stats.getRewardDistribution()
    except Exception as e:
        return str(e)


description = """Gets reward distribution information."""

statsGetRewardDistributionTool = StructuredTool(
    name="gets_reward_distribution",
    description=description,
    func=get_reward_distribution,
    args_schema=ToolInputSchema,
)

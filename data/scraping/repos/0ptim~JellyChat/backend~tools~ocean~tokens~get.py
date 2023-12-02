from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from ..utils import getOcean


class ToolInputSchema(BaseModel):
    token_id: str = Field(..., description="The id of the token")


def get(token_id: str) -> str:
    try:
        return getOcean().tokens.get(token_id)
    except Exception as e:
        return str(e)


description = """Get information about a token with id of the token"""

tokenGetTool = StructuredTool(
    name="get_tokens",
    description=description,
    func=get,
    args_schema=ToolInputSchema,
)

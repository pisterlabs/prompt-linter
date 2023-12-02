from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from ..utils import getOcean


class ToolInputSchema(BaseModel):
    hash: str = Field(..., description="The hash")
    limit: int = Field(..., description="The number of transactions")


def get_transactions(hash: str, limit: int) -> str:
    try:
        return getOcean().blocks.getTransactions(hash, limit)
    except Exception as e:
        return str(e)


description = (
    """Returns all transactions within a block and their corresponding information."""
)

blocksGetTransactionsTool = StructuredTool(
    name="get_transactions",
    description=description,
    func=get_transactions,
    args_schema=ToolInputSchema,
)

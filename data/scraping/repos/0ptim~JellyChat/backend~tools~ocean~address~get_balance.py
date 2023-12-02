from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from ..utils import getOcean


class ToolInputSchema(BaseModel):
    address: str = Field(..., description="The address")


def get_balance(address: str) -> str:
    try:
        return getOcean().address.getBalance(address)
    except Exception as e:
        return str(e)


description = """Gets the current DFI utxo balance of a specific address."""

addressGetBalanceTool = StructuredTool(
    name="get_utxo_balance",
    description=description,
    func=get_balance,
    args_schema=ToolInputSchema,
)

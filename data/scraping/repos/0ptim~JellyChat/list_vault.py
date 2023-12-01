from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from ..utils import getOcean


class ToolInputSchema(BaseModel):
    address: str = Field(..., description="The address")
    limit: int = Field(..., description="Number of vaults to list")


def list_vault(address: str, limit: int) -> str:
    try:
        return getOcean().address.listVault(address=address, size=limit)
    except Exception as e:
        return str(e)


description = """Lists vaults belonging to the specified address"""

addressListVaultTool = StructuredTool(
    name="get_vaults_of_address",
    description=description,
    func=list_vault,
    args_schema=ToolInputSchema,
)

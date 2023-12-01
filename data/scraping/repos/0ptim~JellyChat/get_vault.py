from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from ..utils import getOcean


class ToolInputSchema(BaseModel):
    vault_id: str = Field(..., description="The vault ID")


def get_vault(vault_id: str) -> str:
    try:
        return getOcean().loan.getVault(vault_id)
    except Exception as e:
        return str(e)


description = """Get information about a vault with given vault id."""

loanGetVaultTool = StructuredTool(
    name="get_vault_information",
    description=description,
    func=get_vault,
    args_schema=ToolInputSchema,
)

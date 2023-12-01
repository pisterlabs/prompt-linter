from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from ..utils import getOcean


class ToolInputSchema(BaseModel):
    raw_tx: str = Field(..., description="Raw transaction (hex string)")


def send(raw_tx: str) -> str:
    try:
        return getOcean().rawTx.send(raw_tx)
    except Exception as e:
        return str(e)


description = """Sends the provided raw transaction to the network"""

rawTxSendTool = StructuredTool(
    name="send_raw_transaction",
    description=description,
    func=send,
    args_schema=ToolInputSchema,
)

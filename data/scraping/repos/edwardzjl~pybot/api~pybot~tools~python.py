import asyncio
from typing import Any, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from loguru import logger
from pydantic.v1 import root_validator

from pybot.jupyter import ContextAwareKernelManager, ExecutionRequest, ExecutionResponse
from pybot.tools.base import ExtendedTool


class CodeSandbox(ExtendedTool):
    name = "code_sandbox"
    timeout: int = 60
    """The timeout for the tool in seconds."""
    description = f"""- {name}:
  - Description: {name} is a powerful tool designed for executing Python code, facilitating diverse tasks such as like data analysis, data visualization, etc. {name} will respond with the output of the execution or time out after {timeout} seconds. The driver at '/mnt/data' can be used to save and persist user files. When performing data analysis, inspect the dataset first to make sure you understand it properly.
  - Execution Environment: python3 Jupyter notebook with the following major dependencies:
    - pandas==1.5.3
    - scikit-learn
    - scikit-image
    - seaborn
    - SQLAlchemy
  - Usage Schema: When involking {name}, ensure that you provide a JSON object adhering to the following schema:

    ```yaml
    ToolRequest:
      type: object
      properties:
        tool_name:
          type: string
          enum: ["{name}"]
        tool_input:
          type: string
          description: the code you want {name} to execute
      required: [tool_name, tool_input]
    ```"""
    examples: str = """<|im_start|>system-example-user
{"filename": "test.csv", "path": "/mnt/data/test.csv"}<|im_end|>
<|im_start|>system-example-user
Help me analyze this data.<|im_end|>
<|im_start|>system-example-assistant
Sure, I can help you with that. Let's start by examining the initial rows of the dataset to understand its structure. I'll use the code_sandbox tool for this.
{
    "tool_name": "code_sandbox",
    "tool_input": "import pandas as pd\\n\\n# read the file\\ndf = pd.read_csv(\'/mnt/data/test.csv\')\\n\\n# Display the initial rows of the dataframe\\nprint(df.head())"
}<|im_end|>"""
    gateway_url: str
    kernel_manager: Optional[ContextAwareKernelManager] = None

    @root_validator(pre=True)
    def validate_environment(cls, values: dict[str, Any]) -> dict[str, Any]:
        values["kernel_manager"] = ContextAwareKernelManager(
            gateway_host=values["gateway_url"]
        )
        return values

    def _run(
        self, code: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        kernel = self.kernel_manager.start_kernel()
        with self.kernel_manager.upgrade(str(kernel.id)) as websocket:
            payload = ExecutionRequest.of_code(code)
            logger.debug(f"kernel execution payload: {payload.model_dump_json()}")
            result = ""
            try:
                websocket.send(payload.model_dump_json())
                while message := websocket.recv(timeout=self.timeout):
                    logger.trace(f"kernel execution message: [{message}]")
                    response = ExecutionResponse.model_validate_json(message)
                    match response.msg_type:
                        case "error":
                            result = (
                                f"{response.content.ename}: {response.content.evalue}"
                            )
                            break
                        case "execute_result":
                            result = response.content.data.text_plain
                            break
                        case "stream":
                            result = response.content.text
                            break
                        case _:
                            # debug because we don't handle many message types like status
                            logger.debug(f"Unhandled message type: {response.msg_type}")
            except Exception as e:
                logger.error(f"Something goes wrong, err: {str(e)}")
                result = str(e)
        return result

    async def _arun(
        self, code: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        kernel = await self.kernel_manager.astart_kernel()
        async with self.kernel_manager.aupgrade(str(kernel.id)) as websocket:
            payload = ExecutionRequest.of_code(code)
            logger.debug(f"kernel execution payload: {payload.model_dump_json()}")
            result = ""
            try:
                await websocket.send(payload.model_dump_json())
                while message := await asyncio.wait_for(
                    websocket.recv(), timeout=self.timeout
                ):
                    logger.debug(f"kernel execution message: [{message}]")
                    response = ExecutionResponse.model_validate_json(message)
                    if response.parent_header.msg_id != payload.header.msg_id:
                        # ignore messages not related to this request
                        # should rarely happen, but in case there's some unprocessed messages from previous run
                        logger.debug(
                            f"Ignoring message of parent id {response.parent_header.msg_id} in request {payload.header.msg_id}"
                        )
                        continue
                    match response.msg_type:
                        case "error":
                            result = (
                                f"{response.content.ename}: {response.content.evalue}"
                            )
                        case "execute_result":
                            result = response.content.data.text_plain
                        case "stream":
                            result = response.content.text
                        case "status":
                            if response.content.execution_state == "idle":
                                # idle means the kernel has finished executing
                                # TODO: there will be rare situations that the idle message is received before the execute_result message
                                # See <https://github.com/jupyter-server/enterprise_gateway/blob/54c8e31d9b17418f35454b49db691d2ce5643c22/enterprise_gateway/client/gateway_client.py#L235C9-L235C9>
                                break
                        case _:
                            # debug because we don't handle many message types like status
                            logger.debug(f"Unhandled message type: {response.msg_type}")
            except Exception as e:
                logger.error(f"Something goes wrong, err: {str(e)}")
                result = str(e)
        return result

import time
from typing import Dict, Optional, Union, Any, List
from uuid import UUID, uuid4
import json
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import tracing_enabled
from langchain.llms import OpenAI

class AgentExecutorHandler(BaseCallbackHandler):
    def __init__(self, session_id: str, message_id: str):
        self.session_id = session_id
        self.message_id = message_id
        self.nodes = {}
        self.run_id_to_node_hash = {}
        self.node_hash_to_path = {}

    def set_paths(self, parent_run_id: str, run_id: str, data: Any, *, method=None):
        node_body = {
            "method": method,
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "timestamp": time.time() * 1000,
            "data": data,
        }
        node_hash = hex(hash(json.dumps(node_body)))
        self.nodes[node_hash] = node_body
        self.run_id_to_node_hash[run_id] = node_hash

        parent_path = "{self.session_id}/{self.message_id}"
        if parent_run_id != None:
            found_parent_hash = self.run_id_to_node_hash[parent_run_id]
            parent_path = f"{self.node_hash_to_path[found_parent_hash]}/calls"
        else:
            parent_path = f"{self.session_id}/{self.message_id}/calls"

        current_path = f"{parent_path}/{method}/{node_hash}"
        self.node_hash_to_path[node_hash] = current_path

        print(
            f"\n{len(current_path.split('/')) * '  '} {method} ({str(parent_run_id)[-6:] if parent_run_id else ''} -> {current_path[-6:]}) | {json.dumps(data)}"
        )

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        run_id: str,
        parent_run_id: str,
        **kwargs: Any,
    ) -> Any:
        self.set_paths(parent_run_id, run_id, inputs, method="on_chain_start")

    def on_chain_end(
        self,
        response: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        # print(f"{parent_run_id} -> {run_id} | on_chain_end {response}")
        self.set_paths(parent_run_id, run_id, response, method="on_chain_end")

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        run_id: str,
        parent_run_id: str,
        **kwargs: Any,
    ) -> Any:
        self.set_paths(
            parent_run_id,
            run_id,
            {
                "prompts": prompts,
                "serialized": serialized,
            },
            method="on_llm_start",
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        self.set_paths(
            parent_run_id,
            run_id,
            {"text": response.generations[0][0].text},
            method="on_llm_end",
        )

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        run_id: str,
        parent_run_id: str,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM errors."""
        print(f"on_llm_error {error}")

    # def on_llm_new_token(
    #     self, token: str, run_id: str, parent_run_id: str, **kwargs: Any
    # ) -> Any:
    # print(f"{parent_run_id} -> {run_id} |  on_new_token {token}")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        run_id: str,
        parent_run_id: str,
        **kwargs: Any,
    ) -> Any:
        self.set_paths(
            parent_run_id,
            run_id,
            {"serialized": serialized, "input": input_str},
            method="on_tool_start",
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        self.set_paths(parent_run_id, run_id, {"output": output}, method="on_tool_end")

    def on_tool_error(
        self,
        error: Exception | KeyboardInterrupt,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        self.set_paths(parent_run_id, run_id, error, method="on_tool_error")

    def on_agent_action(
        self, action: AgentAction, run_id: str, parent_run_id: str, **kwargs: Any
    ) -> Any:
        self.set_paths(
            parent_run_id, run_id, action._asdict(), method="on_agent_action"
        )

    def on_agent_finish(
        self, finish: AgentFinish, run_id: str, parent_run_id: str, **kwargs: Any
    ) -> Any:
        self.set_paths(
            parent_run_id, run_id, finish._asdict(), method="on_agent_finish"
        )


# class LLMHandler(BaseCallbackHandler):
# def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
#     print(f"on_new_token {token}")

# def on_llm_start(
#     self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
# ) -> Any:
#     print(f"on_llm_start (I'm the second handler!!) {serialized['name']}")


# # Instantiate the handlers
# handler1 = MyCustomHandlerOne()
# handler2 = MyCustomHandlerTwo()

# # Setup the agent. Only the `llm` will issue callbacks for handler2
# llm = OpenAI(temperature=0, streaming=True, callbacks=[handler2])
# tools = load_tools(["llm-math"], llm=llm)
# agent = initialize_agent(
#     tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
# )

# # Callbacks for handler1 will be issued by every object involved in the
# # Agent execution (llm, llmchain, tool, agent executor)
# agent.run("What is 2 raised to the 0.235 power?", callbacks=[handler1])

"""
Simulates an LLMNode, but does not actually send any data to the LLM.
"""
from promptflow.src.dialogues.node_options import NodeOptions
from promptflow.src.nodes.llm_node import OpenAINode
from promptflow.src.state import State


class DummyNode(OpenAINode):
    """
    Simulates an LLMNode, but does not actually send any data to the LLM.
    """

    dummy_string: str = "dummy string"

    def edit_options(self, event):
        self.options_popup = NodeOptions(
            self.canvas,
            {
                "dummy_string": self.dummy_string,
            },
        )
        self.canvas.wait_window(self.options_popup)
        result = self.options_popup.result
        if self.options_popup.cancelled:
            return
        self.dummy_string = result["dummy_string"]

    def _chat_completion(self, prompt: str, state: State) -> str:
        return self.dummy_string

    def _completion(self, prompt: str, state: State) -> str:
        return self.dummy_string

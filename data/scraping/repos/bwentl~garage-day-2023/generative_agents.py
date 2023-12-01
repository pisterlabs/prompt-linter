import logging
import sys
import os
import warnings

import gradio as gr

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.prompts.prompt import PromptTemplate

from src.util import agent_logs

sys.path.append("./")
from src.models import LlamaModelHandler
from src.docs import DocumentHandler
from src.docs import AggregateRetrieval
from src.agent_executor import AgentExecutorHandler
from src.prompts.tool_select import TOOL_SELECTION_PROMPT


# suppress warnings for demo
warnings.filterwarnings("ignore")
os.environ["PYDEVD_INTERRUPT_THREAD_TIMEOUT"] = "60"
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT "] = "60"
os.environ["MYLANGCHAIN_SAVE_CHAT_HISTORY"] = "0"


class MyLangchainGenerativeAgent:
    """run a generative agent using conversation chain, agent executor, and memory store"""

    template = None
    prompt = None
    conversation = None

    def __init__(self, pipeline, conversation_type="buffer_memory"):
        conversation_type = "buffer_memory"
        # conversation_type = "summary_buffer_memory"
        self.template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        {history}

        ### Instruction:
        Please act as a Socratic questioner. Please respond with a thought-provoking question. All responses should be in the form of a question.

        ### Input:
        {input}

        ### Response:
        """

        self.prompt = PromptTemplate(
            input_variables=["history", "input"], template=self.template
        )
        # Conversation chain
        if conversation_type == "summary_buffer_memory":
            self.conversation = ConversationChain(
                prompt=self.prompt,
                llm=pipeline,
                verbose=True,
                memory=ConversationBufferMemory(
                    human_prefix="### Instruction", ai_prefix="### Response"
                ),
            )
        else:
            self.conversation = ConversationChain(
                prompt=self.prompt,
                llm=pipeline,
                verbose=True,
                memory=ConversationSummaryBufferMemory(
                    llm=pipeline,
                    human_prefix="### Instruction",
                    ai_prefix="### Response",
                ),
            )

        print("conversation chain initiated.")

    def _question_answer_with_memory(self, question):
        langchain_log = agent_logs.read_log()
        answer = self.conversation.predict(input=question)
        return [langchain_log, answer]

    def start_server(self):
        gr.Interface(
            fn=self._question_answer_with_memory,
            inputs=["text"],
            outputs=["textbox", "textbox"],
        ).launch(server_name="0.0.0.0", server_port=7860)
        print("stop")


# Notes on memory implementation (based on Stanford agents)
# https://arxiv.org/abs/2304.03442
#
# requirements
# - context
# - buffer memory (from conversation) into short-term memory
# - offloading of short-term memory into long-term memory (periodically)
# - vectorstore of long-term memory
#
# memory data
# - timestamp
# - last_accessed_timestamp
# - content
# - importance
#
# retrival_score = product of:
# - recency_score = f(current_timestamp,last_accessed_timestamp,exponential_decay_beta)
# - importance = f(llm("please rate the importance of this memory: {content_str}"))
# - relevance = embedding_dist(prompt, content_str)
#
# - get 100 most recent memory
# - calculate retrival_score
# - inject top K (3) memory items into the prompt: "1 days ago - this is memory 1... 3 days ago - this is memory 2 ... 7 days ago - this is memory 3 ...""

if __name__ == "__main__":
    # test this class

    # select model and lora
    model_name = "llama-7b"
    lora_name = "alpaca-lora-7b"

    # Load model
    testAgent = LlamaModelHandler()
    eb = testAgent.get_hf_embedding()
    pipeline, model, tokenizer = testAgent.load_llama_llm(
        model_name=model_name, lora_name=lora_name, max_new_tokens=200
    )

    # Load agent executor
    ## define tool list (excluding any documents)
    test_tool_list = ["wiki", "searx"]

    ## define test documents
    test_doc_info = {
        "examples": {
            "tool_name": "State of Union Document",
            "description": "President Joe Biden's 2023 state of the union address.",
            "files": ["index-docs/examples/state_of_the_union.txt"],
        }
    }

    ## initiate agent executor
    kwarg = {"doc_use_type": "aggregate", "doc_top_k_results": 3}
    test_agent_executor = AgentExecutorHandler(
        pipeline=pipeline,
        embedding=eb,
        tool_names=test_tool_list,
        doc_info=test_doc_info,
        run_tool_selector=True,
        update_long_term_memory=True,
        use_long_term_memory=False,
        **kwarg,
    )

    # Start conversation with initial prompt
    initial_prompt = "Morocco"
    gAgent = MyLangchainGenerativeAgent(pipeline=hf, conversation_type="buffer_memory")
    [log, questioner_thought] = gAgent._question_answer_with_memory(initial_prompt)
    print(questioner_thought)
    agent_exec_response = test_agent_executor.run(questioner_thought)

    # Conversation loop between questioner and agent executor responder
    for i in range(3):
        [log, questioner_thought] = gAgent._question_answer_with_memory(
            agent_exec_response
        )
        print(questioner_thought)

        prev_agent_exec_response_plus_questioner_thought = (
            "Previous response- "
            + agent_exec_response
            + "\nFollow-up question- "
            + questioner_thought
        )
        agent_exec_response = test_agent_executor.run(
            prev_agent_exec_response_plus_questioner_thought
        )
        print(agent_exec_response)

    print("done")

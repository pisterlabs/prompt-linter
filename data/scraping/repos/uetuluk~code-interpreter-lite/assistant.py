from markdown import Markdown
import gradio as gr
from dotenv import load_dotenv
import os
import requests
import io
import re
import base64

import langchain
from langchain import PromptTemplate, LLMChain
from langchain.llms import TextGen
from typing import Any, Dict, List, Optional, Iterator, Tuple
import json
from langchain.schema.output import GenerationChunk
from langchain.callbacks.manager import CallbackManagerForLLMRun
import websocket
from langchain.tools import StructuredTool
from langchain.agents import ZeroShotAgent
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import AgentExecutor
from langchain.schema import AgentAction, LLMResult, AgentFinish, OutputParserException
from threading import Thread
from queue import Queue, Empty

SUPERVISOR_API = "http://localhost:3000"
MODEL_URL = "wss://api.openchat.ritsdev.top"


def setup_assistant():

    class CustomTextGen(TextGen):
        def _get_parameters(self, stop: Optional[List[str]] = None) -> Dict[str, Any]:
            """
            Performs sanity check, preparing parameters in format needed by textgen.

            Args:
                stop (Optional[List[str]]): List of stop sequences for textgen.

            Returns:
                Dictionary containing the combined parameters.
            """

            # Raise error if stop sequences are in both input and default params
            # if self.stop and stop is not None:
            combined_stop = []
            if self.stopping_strings and stop is not None:
                # combine
                combined_stop = self.stopping_strings + stop
                # raise ValueError("`stop` found in both the input and default params.")

            if self.preset is None:
                params = self._default_params
            else:
                params = {"preset": self.preset}

            # then sets it as configured, or default to an empty list:
            params["stop"] = combined_stop or self.stopping_strings or stop or []

            return params

        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> str:
            """Call the textgen web API and return the output.

            Args:
                prompt: The prompt to use for generation.
                stop: A list of strings to stop generation when encountered.

            Returns:
                The generated text.

            Example:
                .. code-block:: python

                    from langchain.llms import TextGen
                    llm = TextGen(model_url="http://localhost:5000")
                    llm("Write a story about llamas.")
            """
            if self.streaming:
                combined_text_output = ""
                for chunk in self._stream(
                    prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
                ):
                    combined_text_output += chunk.text
                print(prompt + combined_text_output)
                result = combined_text_output

            else:
                url = f"{self.model_url}/api/v1/generate"
                params = self._get_parameters(stop)
                request = params.copy()
                request["prompt"] = prompt
                response = requests.post(url, json=request)

                if response.status_code == 200:
                    result = response.json()["results"][0]["text"]
                    print(prompt + result)
                else:
                    print(f"ERROR: response: {response}")
                    result = ""

            return result

        def _stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> Iterator[GenerationChunk]:
            """Yields results objects as they are generated in real time.

            It also calls the callback manager's on_llm_new_token event with
            similar parameters to the OpenAI LLM class method of the same name.

            Args:
                prompt: The prompts to pass into the model.
                stop: Optional list of stop words to use when generating.

            Returns:
                A generator representing the stream of tokens being generated.

            Yields:
                A dictionary like objects containing a string token and metadata.
                See text-generation-webui docs and below for more.

            Example:
                .. code-block:: python

                    from langchain.llms import TextGen
                    llm = TextGen(
                        model_url = "ws://localhost:5005"
                        streaming=True
                    )
                    for chunk in llm.stream("Ask 'Hi, how are you?' like a pirate:'",
                            stop=["'","\n"]):
                        print(chunk, end='', flush=True)

            """
            params = {**self._get_parameters(stop), **kwargs}

            url = f"{self.model_url}/api/v1/stream"

            request = params.copy()
            request["prompt"] = prompt

            websocket_client = websocket.WebSocket()

            websocket_client.connect(url)

            websocket_client.send(json.dumps(request))

            while True:
                result = websocket_client.recv()
                result = json.loads(result)

                if result["event"] == "text_stream":
                    chunk = GenerationChunk(
                        text=result["text"],
                        generation_info=None,
                    )
                    yield chunk
                elif result["event"] == "stream_end":
                    websocket_client.close()
                    return

                if run_manager:
                    run_manager.on_llm_new_token(token=chunk.text)

    token_response = requests.post(
        f'{SUPERVISOR_API}/token', data={'username': "a", 'password': "a"}, timeout=600)

    token_text = token_response.json().get('access_token')
    token_instance = token_text

    container_response = requests.post(
        f'{SUPERVISOR_API}/container', headers={
            'Authorization': f'Bearer {token_text}'
        }, timeout=600
    )

    def code_interpreter_lite(code: str) -> str:
        """Execute the python code and return the result."""
        # handle markdown
        def extract_code_from_markdown(md_text):
            # Using regex to extract text between ```
            pattern = r"```[\w]*\n(.*?)```"
            match = re.search(pattern, md_text, re.DOTALL)
            if match:
                return match.group(1).strip()
            else:
                # might not be markdown
                return md_text
        code = extract_code_from_markdown(code)

        code_response = requests.post(
            f'{SUPERVISOR_API}/run', headers={
                'Authorization': f'Bearer {token_instance}'
            }, json={
                'code_string': code
            }, timeout=600)
        if code_response.status_code != 200:
            raise Exception("No container created yet", code_response.text)
            # return {
            #     chatbot: chatbot_instance,
            #     msg: msg_instance,
            # }
        result = code_response.json()

        def is_base64(string):
            try:
                # Try to decode the string as base64
                base64.b64decode(string, validate=True)
                return True
            except:
                return False

        # handle base64 results - ie images
        print("Result from tool:", result)
        if len(result) > 1024:
            result = "The result is too long to display."

        return result

    tool = StructuredTool.from_function(
        func=code_interpreter_lite, name="CIL", description="useful for running python code. The input should be a string of python code.")

    tools = [tool]

    prefix = """<|im_start|>system
You are an assistant to a user who is trying to solve a question. You can write and execute Python code to find the solution.
You should only use the name of the tool to call it.
If you need to output any kind of graph to the user, you should save it in a file and return the file location.
You have access to the following tools:"""
    suffix = """Begin! Remember to use the tools with the correct format which is:
Action: CIL
Action Input: ```python
your code
```<|im_end|>
<|im_start|>user
Question: {input}<|im_end|>
<|im_start|>assistant
Thought: {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools=[tool], prefix=prefix, suffix=suffix, input_variables=[
            "input", "agent_scratchpad"]
    )

    model_url = MODEL_URL

    llm = CustomTextGen(model_url=model_url, temperature=0.1, max_new_tokens=1024, streaming=True, callbacks=[
                        StreamingStdOutCallbackHandler()], stopping_strings=["<|im_end|>", "<|im_sep|>", "Observation:"])

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)

    agent_executor_instance = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, return_intermediate_steps=True, max_iterations=2
    )

    return agent_executor_instance


agent_executor = setup_assistant()

# chatbot style

ALREADY_CONVERTED_MARK = "<!-- ALREADY CONVERTED BY PARSER. -->"


def insert_newline_before_triple_backtick(text):
    modified_text = text.replace(" ```", "\n```")

    return modified_text


def insert_summary_block(text):
    pattern = r"(Action: CIL)(.*?)(Observation:|$)"
    replacement = r"<details><summary>CIL Code</summary>\n\1\2\n</details>\n\3"

    return re.sub(pattern, replacement, text, flags=re.DOTALL)


def postprocess(
    self, history: List[Tuple[str | None, str | None]]
) -> List[Tuple[str | None, str | None]]:
    markdown_converter = Markdown(
        extensions=["nl2br", "fenced_code"])

    if history is None or history == []:
        return []

    print(history)

    formatted_history = []
    for conversation in history:
        user, bot = conversation

        if user == None or user.endswith(ALREADY_CONVERTED_MARK):
            formatted_user = user
        else:
            formatted_user = markdown_converter.convert(
                user) + ALREADY_CONVERTED_MARK

        if bot == None or bot.endswith(ALREADY_CONVERTED_MARK):
            formatted_bot = bot
        else:
            preformatted_bot = insert_newline_before_triple_backtick(bot)
            summary_bot = insert_summary_block(preformatted_bot)
            print(summary_bot)

            formatted_bot = markdown_converter.convert(
                summary_bot) + ALREADY_CONVERTED_MARK

        formatted_history.append((formatted_user, formatted_bot))

    return formatted_history


gr.Chatbot.postprocess = postprocess

with gr.Blocks() as demo:

    with gr.Column() as chatbot_column:
        chatbot = gr.Chatbot()
        with gr.Row() as chatbot_input:
            with gr.Column():
                msg = gr.Textbox(placeholder="Type your message here")
            with gr.Column():
                send = gr.Button(value="Send", variant="primary")
                regenerate = gr.Button(
                    value="Regenerate", variant="secondary", interactive=False)

    def message_handle(chatbot_instance, msg_instance):
        return {
            chatbot: chatbot_instance + [[msg_instance, None]],
            msg: "",
            regenerate: gr.update(interactive=True),
        }

    def regenerate_message_handle(chatbot_instance):
        previous_message = chatbot_instance[-1][0]
        chatbot_instance[-1] = [previous_message, None]

        return {
            chatbot: chatbot_instance,
        }

    def chatbot_handle(chatbot_instance):

        # class ChatbotHandler(BaseCallbackHandler):
        #     def __init__(self):
        #         self.chatbot_response = ""
        #         super().__init__()

        #     def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        #         self.chatbot_response += outputs.get("output", "") + '\n'

        #     def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        #         self.chatbot_response += f'```\n{output}\n```\n'

        #     def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        #         chatbot_thought = action.log.split("\n")[0]
        #         chatbot_thought = chatbot_thought.replace("Thought: ", "")

        #         if isinstance(action.tool_input, str):
        #             chatbot_tool_input_code_string = action.tool_input
        #         else:
        #             chatbot_tool_input_code_string = action.tool_input.get(
        #                 "code")
        #         self.chatbot_response += f"{chatbot_thought}\n"
        #         self.chatbot_response += f'```\n{chatbot_tool_input_code_string}\n```\n'

        #     def get_chatbot_response(self):
        #         return self.chatbot_response

        class QueueCallback(BaseCallbackHandler):
            """Callback handler for streaming LLM responses to a queue."""

            def __init__(self, queue):
                self.queue = queue

            def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
                self.queue.put(token)

            def on_tool_end(self, output: str, **kwargs: Any) -> None:
                self.queue.put(f'Observation: \n```\n{output}\n```\n')

            def on_llm_end(self, *args, **kwargs: Any) -> None:
                return self.queue.empty()

        streaming_queue = Queue()
        job_done = object()

        user_message = chatbot_instance[-1][0]

        def task():
            try:
                agent_executor(
                    user_message, callbacks=[QueueCallback(streaming_queue)])
                streaming_queue.put(job_done)
            except OutputParserException as error:
                streaming_queue.put(job_done)
                raise gr.Error(
                    "Assistant could not handle the request. Error: " + str(error))

        streaming_thread = Thread(target=task)
        streaming_thread.start()

        chatbot_instance[-1][1] = ""

        while True:
            try:
                next_token = streaming_queue.get(True, timeout=1)
                if next_token is job_done:
                    break
                chatbot_instance[-1][1] += next_token
                yield chatbot_instance
            except Empty:
                continue

    send.click(message_handle, [chatbot, msg], [
        chatbot, msg, regenerate]).then(
        chatbot_handle, [chatbot], [chatbot]
    )
    regenerate.click(regenerate_message_handle, [chatbot], [
        chatbot]).then(
        chatbot_handle, [chatbot], [chatbot]
    )


demo.queue()
demo.launch(server_port=7861)

import os
import threading
from typing import Any, Dict, List, Sequence, Union

from slack_sdk.web import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.socket_mode.request import SocketModeRequest
from agent.factory import AgentFactory
from listeners.agent_listener import AgentListener
from langchain.callbacks import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult


class SlackListener(AgentListener):
    def __init__(self, factory: AgentFactory):
        # Initialize SocketModeClient with an app-level token + WebClient
        self.client = SocketModeClient(
            # This app-level token will be used only for establishing a connection
            app_token=os.environ.get("SLACK_APP_TOKEN"),  # xapp-A111-222-xyz
            # You will be using this WebClient for performing Web API calls in listeners
            web_client=WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))  # xoxb-111-222-xyz
        
        )
        super().__init__(factory)

            
    def on_message(self, req: SocketModeRequest, message: str):
        # reply to the message with an acknowledgement   
        send_slack_text_message(self.client, req, "Let me think about that...")
        handler = SlackCallbackHandler(self.client, req)
        agent = self.factory.new_k8s_engineer([handler])
        agent({"input": message})
        del agent

    def listen(self, interrupt: threading.Event):
        self.client.socket_mode_request_listeners.append(self.process)
        self.client.connect()
        interrupt.wait()

    def start(self) -> threading.Event:
    # run listen in a separate thread
        interrupt = threading.Event()
        thread = threading.Thread(target=self.listen, args=(interrupt,))
        thread.start()
        return interrupt

    # TODO: Find a way to not handle all requests, but only the ones that are relevant to the bot
    def process(self, client: SocketModeClient, req: SocketModeRequest):
        if req.type == "events_api":
            # Acknowledge the request anyway
            response = SocketModeResponse(envelope_id=req.envelope_id)
            client.send_socket_mode_response(response)

            # Add a reaction to the message if it's a new message
            if req.payload["event"]["type"] == "message" or req.payload["event"]["type"] == "app_mention"\
                and req.payload["event"].get("subtype") is None:
                # ignore messages from the bot itself
                if req.payload["event"]["user"] == client.web_client.auth_test()["user_id"]:
                    return
                
                message = req.payload["event"]["text"]
                # remove the bot mention
                if req.payload["event"]["type"] == "app_mention":
                    message = message.split(" ", 1)[1]
                threading.Thread(target=self.on_message(req, message)).start()

class SlackThread:
    thread_ts: str = None
    channel: str = None
    lock: threading.Lock

    def __init__(self, channel: str, thread_ts: str):
        self.channel = channel
        self.thread_ts = thread_ts
        self.lock = threading.Lock()
                

def send_slack_text_message(client: SocketModeClient, req: SocketModeRequest, message: str):
        client.web_client.chat_postMessage(
            channel=req.payload["event"]["channel"],
            thread_ts=req.payload["event"]["ts"],
            text=message)
        
def send_slack_block_message(client: SocketModeClient, req: SocketModeRequest, blocks: Sequence[dict]):
    client.web_client.chat_postMessage(
        channel=req.payload["event"]["channel"],
        thread_ts=req.payload["event"]["ts"],
        blocks=blocks)

def parse_intermediate_steps_into_slack_message(steps) -> dict:
    """steps is a NamedTuple with fields that looks like this:
    [(AgentAction(tool='Search', tool_input='Leo DiCaprio girlfriend', log=' I should look up who Leo DiCaprio is dating\nAction: Search\nAction Input: "Leo DiCaprio girlfriend"'), 'Camila Morrone'), (AgentAction(tool='Search', tool_input='Camila Morrone age', log=' I should look up how old Camila Morrone is\nAction: Search\nAction Input: "Camila Morrone age"'), '25 years'), (AgentAction(tool='Calculator', tool_input='25^0.43', log=' I should calculate what 25 years raised to the 0.43 power is\nAction: Calculator\nAction Input: 25^0.43'), 'Answer: 3.991298452658078\n')]
    """
    message = {}
    message['blocks'] = []
    for step in steps:
        message['blocks'].extend(create_final_block(step))
    return message

def create_final_block(step) -> Sequence[dict]:
    """step is a NamedTuple with fields that looks like this:
    (AgentAction(tool='Search', tool_input='Leo DiCaprio girlfriend', log=' I should look up who Leo DiCaprio is dating\nAction: Search\nAction Input: "Leo DiCaprio girlfriend"'), 'Camila Morrone')
    """
    block = []
    block.extend(create_action_block(step[0]))
    block.extend(create_output_block(step[1]))
    return block

def create_action_block(action: AgentAction) -> Sequence[dict]:
    """action is a NamedTuple with fields that looks like this:
    AgentAction(tool='Search', tool_input='Leo DiCaprio girlfriend', log=' I should look up who Leo DiCaprio is dating\nAction: Search\nAction Input: "Leo DiCaprio girlfriend"')
    """
    block = []
    tool = action.tool.replace("\\n", "\n")
    block.append({
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f'Action: {tool}'
        }
    })
    tool_input = action.tool_input.replace("\\n", "\n")
    block.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f'*Action Input:*\n{tool_input}'
        }
    })
    return block

def create_output_block(answer: str) -> Sequence[dict]:
    """answer is a string that looks like this:
    'Camila Morrone'
    """
    block = []
    answer = answer.replace("\\n", "\n")
    block.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f'*Answer:*\n```{answer}```'
        }
    })
    return block

class SlackCallbackHandler(BaseCallbackHandler):
    client: SocketModeClient
    req: SocketModeRequest
    def __init__(self, client: SocketModeClient, req: SocketModeRequest):
        self.client = client
        self.req = req

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        message_block = create_output_block(output)
        send_slack_block_message(self.client, self.req, message_block)

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""

    
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        message_block = create_action_block(action)
        send_slack_block_message(self.client, self.req, message_block)

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        message_block = create_output_block(finish.log)
        send_slack_block_message(self.client, self.req, message_block)

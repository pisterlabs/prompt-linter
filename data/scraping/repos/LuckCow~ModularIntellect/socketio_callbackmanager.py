import logging
from typing import Any, Dict, List, Union
from abc import ABC, abstractmethod
from threading import Event

from flask import jsonify
from flask_socketio import SocketIO
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult, AgentAction, AgentFinish

from src.web.chain_state import ChainState

logger = logging.getLogger(__name__)

class SocketIOCallbackHandler(BaseCallbackHandler):
    def __init__(self, socketio: SocketIO, room: str):
        self.socketio = socketio
        self.room = room
        self.chain_state = ChainState()

    def chain_execution_state(self):
        #print('chain_execution_state: ', self.chain_state.chain_blocks)
        return jsonify(
            chainBlocks=[{
                'title': block.title,
                'inputs': block.inputs,
                'outputs': block.outputs
            } for block in self.chain_state.chain_blocks],
            currentBlockIndex=self.chain_state.current_block_index,
        )

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        logging.info('on_chain_start: serialized: %s, inputs: %s', serialized, inputs)
        self.chain_state.add_chain_block('Chain Title Placeholder', inputs)
        self.socketio.emit('chain_start', {'serialized': serialized, 'inputs': inputs})#, room=self.room)

        # Create a socket event listener for input updates
        @self.socketio.on("input_update")
        def handle_input_update(data):
            blockIndex = data["blockIndex"]
            key = data["key"]
            value = data["value"]

            # Janky pass by reference
            inputs[key] = value

        # Create an Event to wait for user confirmation
        chain_start_confirm_event = Event()

        # Create callback function to continue execution
        @self.socketio.on('chain_start_confirm')
        def chain_start_confirm_callback():
            logging.info('chain_start_confirm_callback')
            chain_start_confirm_event.set()

        # Wait for the event to be set by the frontend's confirmation
        chain_start_confirm_event.wait()

        # Remove the event listener to avoid memory leaks
        #self.socketio.off('chain_start_confirm', chain_start_confirm_callback, room=self.room)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        logging.info('on_chain_end: outputs: %s', outputs)
        self.chain_state.set_chain_block_outputs(outputs)
        self.socketio.emit('chain_end', {'outputs': outputs})#, room=self.room)

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        self.socketio.emit('llm_start', {'serialized': serialized, 'prompts': prompts}, room=self.room)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        self.socketio.emit('llm_new_token', {'token': token}, room=self.room)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        self.socketio.emit('llm_end', {'response': response.json()}, room=self.room)

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        self.socketio.emit('llm_error', {'error': str(error)}, room=self.room)

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        self.socketio.emit('chain_error', {'error': str(error)}, room=self.room)

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        self.socketio.emit('tool_start', {'serialized': serialized, 'input_str': input_str}, room=self.room)

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        self.socketio.emit('tool_end', {'output': output}, room=self.room)

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        self.socketio.emit('tool_error', {'error': str(error)}, room=self.room)

    def on_text(self, text: str, **kwargs: Any) -> Any:
        self.socketio.emit('text', {'text': text}, room=self.room)

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        self.socketio.emit('agent_action', {'action': action.log}, room=self.room)

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        self.socketio.emit('agent_finish', {'finish': finish.log}, room=self.room)

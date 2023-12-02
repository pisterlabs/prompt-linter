# This chat variant determines if the user's query is related to a widget or a search
import re
import time
import uuid
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, Generator, List, Optional, Union, Literal, TypedDict, Callable

from gpt_index.utils import ErrorToRetry, retry_on_exceptions_with_backoff
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import context
import utils
import utils.timing as timing
from utils import error_wrap, ensure_wallet_connected, ConnectedWalletRequired, FetchError, ExecError
from utils.constants import WIDGET_INFO_TOKEN_LIMIT
from utils.common import modelname_to_contextsize
import registry
import streaming
from chat.container import ContainerMixin, dataclass_to_container_params
from .base import (
    BaseChat, ChatHistory, Response, ChatOutputParser,
)
from integrations import (
    etherscan, defillama, center, opensea,
)
from ui_workflows import (
    aave, ens
)
from ui_workflows.multistep_handler import register_ens_domain, exec_aave_operation
from tools.index_widget import *
from finetune.dataset import (
    format_widgets_for_prompt,
    TEMPLATE,
    STOP,
    NO_WIDGET_TOKEN,
)


# MODEL_NAME = 'curie:ft-yield-inc-2023-05-30-20-19-41'
MODEL_NAME = 'curie:ft-yield-inc:truncate-task-info-2023-06-01-00-37-29'
MAX_TOKENS = 200


@registry.register_class
class FineTunedChat(BaseChat):
    def __init__(self, widget_index: Any, top_k: int = 3, fallback_chat: Optional[BaseChat] = None, model_name: Optional[str] = "text-davinci-003", evaluate_widgets: bool = True) -> None:
        super().__init__()
        self.output_parser = ChatOutputParser()
        self.widget_prompt = PromptTemplate(
            input_variables=["task_info", "chat_history", "user_input"],
            template=TEMPLATE,
            output_parser=self.output_parser,
        )
        self.widget_index = widget_index
        self.top_k = top_k
        self.fallback_chat = fallback_chat
        self.model_name = model_name or MODEL_NAME
        self.evaluate_widgets = evaluate_widgets
        self.token_limit = max(1800, modelname_to_contextsize(model_name) - WIDGET_INFO_TOKEN_LIMIT)

    def receive_input(
            self,
            history: ChatHistory,
            userinput: str,
            send: Callable,
            message_id: Optional[uuid.UUID] = None,
            before_message_id: Optional[uuid.UUID] = None,
    ) -> None:
        userinput = userinput.strip()
        history_string = history.to_string(system_prefix=None, token_limit=self.token_limit, before_message_id=before_message_id)  # omit system messages

        history.add_user_message(userinput, message_id=message_id, before_message_id=before_message_id)
        timing.init()

        bot_chat_message_id = None
        bot_response = ''
        has_sent_bot_response = False

        def bot_flush(response):
            nonlocal bot_chat_message_id
            response = response.strip()
            send(Response(
                response=response,
                still_thinking=False,
                actor='bot',
                operation='replace',
            ), last_chat_message_id=bot_chat_message_id, before_message_id=before_message_id)
            history.add_bot_message(response, message_id=bot_chat_message_id, before_message_id=before_message_id)

        def bot_new_token_handler(token):
            nonlocal bot_chat_message_id, bot_response, has_sent_bot_response

            bot_response += token
            if not bot_response.strip():
                # don't start returning something until we have the first non-whitespace char
                return

            timing.log('first_visible_bot_token')
            bot_chat_message_id = send(Response(
                response=token,
                still_thinking=False,
                actor='bot',
                operation='append' if bot_chat_message_id is not None else 'create',
            ), last_chat_message_id=bot_chat_message_id, before_message_id=before_message_id)
            has_sent_bot_response = True

        new_token_handler = bot_new_token_handler
        response_buffer = ""

        def injection_handler(token):
            nonlocal new_token_handler, response_buffer

            timing.log('first_token')
            timing.log('first_widget_token')  # for comparison with basic agent

            response_buffer += token
            while WIDGET_START in response_buffer and self.evaluate_widgets:
                if WIDGET_END in response_buffer:
                    # parse fetch command
                    response_buffer = iterative_evaluate(response_buffer)
                    if isinstance(response_buffer, Callable):  # handle delegated streaming
                        def handler(token):
                            nonlocal new_token_handler
                            timing.log('first_visible_widget_response_token')
                            return new_token_handler(token)
                        response_buffer(handler)
                        response_buffer = ""
                        return
                    elif isinstance(response_buffer, Generator):  # handle stream of widgets
                        for item in response_buffer:
                            timing.log('first_visible_widget_response_token')
                            new_token_handler(str(item) + "\n")
                        response_buffer = ""
                        return
                    elif len(response_buffer.split(WIDGET_START)) == len(response_buffer.split(WIDGET_END)):
                        # matching pairs of open/close, just flush
                        # NB: for better frontend parsing of nested widgets, we need an invariant that
                        # there are no two independent widgets on the same line, otherwise we can't
                        # detect the closing tag properly when there is nesting.
                        response_buffer = response_buffer.replace(WIDGET_END, WIDGET_END + '\n')
                        break
                    else:
                        # keep waiting
                        return
                else:
                    # keep waiting
                    return

            if len(response_buffer) < len(NO_WIDGET_TOKEN) and NO_WIDGET_TOKEN.startswith(response_buffer):
                # keep waiting
                return
            elif response_buffer.startswith(NO_WIDGET_TOKEN):
                # don't emit this in the stream, we will handle the final response below
                return
            elif 0 < len(response_buffer) < len(WIDGET_START) and WIDGET_START.startswith(response_buffer):
                # keep waiting if we could potentially be receiving WIDGET_START
                return
            token = response_buffer
            response_buffer = ""
            if token.strip():
                timing.log('first_visible_widget_response_token')
            new_token_handler(token)

        if self.widget_index is None:
            task_info = ""
        else:
            widgets = retry_on_exceptions_with_backoff(
                lambda: self.widget_index.similarity_search(userinput, k=self.top_k),
                [ErrorToRetry(TypeError)],
            )
            timing.log('widget_index_lookup_done')
            # task_info = '\n'.join([f'Widget: {widget.page_content}' for widget in widgets])
            task_info = format_widgets_for_prompt(widgets)

        example = {
            "task_info": task_info,
            "chat_history": history_string,
            "user_input": userinput,
            "stop": [STOP],
        }

        chain = streaming.get_streaming_chain(self.widget_prompt, injection_handler, model_name=self.model_name, max_tokens=MAX_TOKENS)

        with context.with_request_context(history.wallet_address, message_id):
            result = chain.run(example).strip()

        if result == NO_WIDGET_TOKEN:
            if self.fallback_chat is not None:
                # call the fallback chat
                self.fallback_chat._inner_receive_input(
                    history,
                    history_string,
                    userinput,
                    send,
                    message_id=message_id,
                    before_message_id=before_message_id,
                )
            else:
                bot_chat_message_id = send(Response(
                    response="I'm sorry, I don't understand. Please try again.",
                    still_thinking=False,
                    actor='bot',
                    operation='replace' if bot_chat_message_id is not None else 'create',
                ), last_chat_message_id=bot_chat_message_id, before_message_id=before_message_id)
            return

        timing.log('response_done')

        if bot_chat_message_id is not None:
            bot_flush(bot_response)

        response = f'Timings - {timing.report()}'
        system_chat_message_id = send(Response(response=response, actor='system'), before_message_id=before_message_id)
        history.add_system_message(response, message_id=system_chat_message_id, before_message_id=before_message_id)

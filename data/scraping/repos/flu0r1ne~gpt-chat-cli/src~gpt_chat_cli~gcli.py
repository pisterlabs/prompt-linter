#!/bin/env python3

import sys
import openai
import pickle
import os
import datetime

from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Optional

from .openai_wrappers import (
    create_chat_completion,
    list_models,
    OpenAIChatResponse,
    OpenAIChatResponseStream,
    FinishReason,
    Role,
    ChatMessage
)

from .argparsing import (
    parse_args,
    Arguments,
    DisplayArguments,
    CompletionArguments,
    DebugArguments,
    MessageSource
)

from .version import VERSION
from .color import get_color_codes
from .chat_colorizer import ChatColorizer

###########################
####      UTILS        ####
###########################

def resolve_initial_message(src: MessageSource, interactive=False) -> str:
    msg = None

    if src.message:
        msg = src.message
    elif src.prompt_from_fd:
        with os.fdopen(src.prompt_from_fd, "r") as f:
            msg = f.read()
    elif src.prompt_from_file:
        with open(src.prompt_from_file, "r") as f:
            msg = f.read()
    elif not interactive:
        msg = sys.stdin.read()

    return msg

def get_system_message(system_message : Optional[str]):

    if not system_message:

        current_date_time = datetime.datetime.now()

        system_message = f'The current date is {current_date_time}. When emitting code or producing markdown, ensure to label fenced code blocks with the language in use.'

    return ChatMessage(Role.SYSTEM, system_message)

def enable_emacs_editing():
    try:
        import readline
    except ImportError:
        pass

###########################
####   SAVE / REPLAY   ####
###########################

@dataclass
class CompletionContext:
    message: str
    completion_args: CompletionArguments
    system_message: Optional[str] = None

def create_singleton_chat_completion(ctx : CompletionContext):

    hist = [
        get_system_message(ctx.system_message),
        ChatMessage(Role.USER, ctx.message)
    ]

    completion = create_chat_completion(hist, ctx.completion_args)

    return completion

def save_response_and_arguments(args : Arguments) -> None:

    message = resolve_initial_message(args.initial_message)

    ctx = CompletionContext(
        message=message,
        completion_args=args.completion_args,
        system_message=args.system_message
    )

    completion = create_singleton_chat_completion(
        message,
        args.completion_args,
        args.system_message,
    )

    completion = list(completion)

    filename = args.debug_args.save_response_to_file

    with open(filename, 'wb') as f:
        pickle.dump((ctx, completion,), f)

def load_response_and_arguments(args : Arguments) \
        -> Tuple[CompletionContext, OpenAIChatResponseStream]:

    filename = args.debug_args.load_response_from_file

    with open(filename, 'rb') as f:
        ctx, completion = pickle.load(f)

    return (ctx, completion)

#########################
#### PRETTY PRINTING ####
#########################

@dataclass
class CumulativeResponse:
    delta_content: str = ""
    finish_reason: FinishReason = FinishReason.NONE
    content: str = ""

    def take_delta(self : "CumulativeResponse"):
        chunk = self.delta_content
        self.delta_content = ""
        return chunk

    def add_content(self : "CumulativeResponse", new_chunk : str):
        self.content += new_chunk
        self.delta_content += new_chunk

def print_streamed_response(
        display_args : DisplayArguments,
        completion : OpenAIChatResponseStream,
        n_completions : int,
        return_responses : bool = False
    ) -> None:
    """
    Print the response in real time by printing the deltas as they occur. If multiple responses
    are requested, print the first in real-time, accumulating the others in the background. One the
    first response completes, move on to the second response printing the deltas in real time. Continue
    on until all responses have been printed.
    """

    no_color = not display_args.color

    COLOR_CODE = get_color_codes(no_color = no_color)
    adornments = display_args.adornments

    cumu_responses = defaultdict(CumulativeResponse)
    display_idx = 0
    prompt_printed = False

    chat_colorizer = ChatColorizer(no_color = no_color)

    for update in completion:

        for choice in update.choices:
            delta = choice.delta

            if delta.content:
                cumu_responses[choice.index].add_content(delta.content)

            if choice.finish_reason is not FinishReason.NONE:
                cumu_responses[choice.index].finish_reason = choice.finish_reason

        display_response = cumu_responses[display_idx]

        if not prompt_printed and adornments:
            res_indicator = '' if n_completions == 1 else \
                    f' {display_idx + 1}/{n_completions}'
            PROMPT = f'[{COLOR_CODE.GREEN}{update.model}{COLOR_CODE.RESET}{COLOR_CODE.RED}{res_indicator}{COLOR_CODE.RESET}]'
            prompt_printed = True
            print(PROMPT, end=' ', flush=True)

        content = display_response.take_delta()
        chat_colorizer.add_chunk( content )

        chat_colorizer.print()

        if display_response.finish_reason is not FinishReason.NONE:
            chat_colorizer.finish()
            chat_colorizer.print()
            chat_colorizer = ChatColorizer( no_color=no_color )

            if display_idx < n_completions:
                display_idx += 1
                prompt_printed = False

            if adornments:
                print(end='\n\n', flush=True)
            else:
                print(end='\n', flush=True)

    if return_responses:
        return [ cumu_responses[i].content for i in range(n_completions) ]

#########################
####    COMMANDS     ####
#########################

def cmd_version():
    print(f'version {VERSION}')

def cmd_list_models():
    for model in list_models():
        print(model)

def surround_ansi_escapes(prompt, start = "\x01", end = "\x02"):
        '''
        Fixes issue on Linux with the readline module
        See: https://github.com/python/cpython/issues/61539
        '''
        escaped = False
        result = ""

        for c in prompt:
                if c == "\x1b" and not escaped:
                        result += start + c
                        escaped = True
                elif c.isalpha() and escaped:
                        result += c + end
                        escaped = False
                else:
                        result += c

        return result

def cmd_interactive(args : Arguments):

    enable_emacs_editing()

    COLOR_CODE = get_color_codes(no_color = not args.display_args.color)

    completion_args = args.completion_args
    display_args = args.display_args

    hist = [ get_system_message( args.system_message ) ]

    PROMPT = f'[{COLOR_CODE.WHITE}#{COLOR_CODE.RESET}] '
    PROMPT = surround_ansi_escapes(PROMPT)

    def prompt_message() -> bool:

        # Control-D closes the input stream
        try:
            message = input( PROMPT )
        except (EOFError, KeyboardInterrupt):
            print()
            return False

        hist.append( ChatMessage( Role.USER, message ) )

        return True

    print(f'GPT Chat CLI version {VERSION}')
    print(f'Press Control-D to exit')

    initial_message = resolve_initial_message(args.initial_message, interactive=True)

    if initial_message:
        print( PROMPT, initial_message, sep='', flush=True )
        hist.append( ChatMessage( Role.USER, initial_message ) )
    else:
        if not prompt_message():
            return

    while True:

        completion = create_chat_completion(hist, completion_args)

        try:
            response = print_streamed_response(
                display_args, completion, 1, return_responses=True,
            )[0]

            hist.append( ChatMessage(Role.ASSISTANT, response) )
        except KeyboardInterrupt:
            print()

        if not prompt_message():
            return

def cmd_singleton(args: Arguments):
    completion_args = args.completion_args

    debug_args : DebugArguments = args.debug_args
    message = args.initial_message

    if debug_args.save_response_to_file:
        save_response_and_arguments(args)
        return
    elif debug_args.load_response_from_file:
        ctx, completion = load_response_and_arguments(args)

        message = ctx.message
        completion_args = ctx.completion_args
    else:
        # message is only None is a TTY is not attached
        message = resolve_initial_message(args.initial_message)

        ctx = CompletionContext(
            message=message,
            completion_args=completion_args,
            system_message=args.system_message
        )

        completion = create_singleton_chat_completion(ctx)

    print_streamed_response(
        args.display_args,
        completion,
        completion_args.n_completions
    )


def main():
    args = parse_args()

    if args.version:
        cmd_version()
    elif args.list_models:
        cmd_list_models()
    elif args.interactive:
        cmd_interactive(args)
    else:
        cmd_singleton(args)

if __name__ == "__main__":
    main()

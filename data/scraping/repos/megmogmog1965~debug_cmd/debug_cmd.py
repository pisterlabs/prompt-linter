#!/usr/bin/env python3
# coding: utf-8

# ----------------------------------------------------------------------------
# Created By  : Yusuke Kawatsu
# Created Date: 2023/05/31
# Usage       : python3 debug_cmd.py <linux command separated by spaces>
#               python3 debug_cmd.py -c '<linux command on shell>'
# Description : Debug linux command error by using GPT/LLM.
# ---------------------------------------------------------------------------

import os
import sys
import locale
import argparse
import platform
import subprocess
from typing import Generator, Union

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter


# CLI encoding.
_CLI_ENCODING = locale.getpreferredencoding()

# llm.
# :see: https://platform.openai.com/docs/models/model-endpoint-compatibility
# _LLM: ChatOpenAI = ChatOpenAI(temperature=1, model_name="gpt-4", max_tokens=1024 * 4)
_LLM: ChatOpenAI = ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo-16k", max_tokens=1024 * 8)


def main() -> None:
    """
    Entry point.
    """
    _assert_environment_variables()
    commands, force = _load_cli_args()

    return_code, cmd_out = _exec_command(commands)

    # successfully ends.
    if return_code == 0 and not force:
        print('\n-------- コマンドは成功しました --------')
        return

    # ask llm about error.
    print('\n-------- エラー原因を解析中 --------')
    cmd_str = ' '.join(commands) if type(commands) is list else commands
    answer = _ask_llm_about_error(cmd=cmd_str, return_code=return_code, error_message=cmd_out)
    print(answer)


def _assert_environment_variables() -> None:
    """
    ensure essential environment variables.
    """
    environments = ['OPENAI_API_KEY']

    for e in environments:
        if e not in os.environ:
            sys.stderr.write(f'Please set environment variable "{e}".')
            sys.exit(1)


def _load_cli_args() -> tuple[Union[str, list[str]], bool]:
    """
    Parse CLI arguments.

    :return: (<command line arguments>, <-f option>).
    """
    parser = argparse.ArgumentParser(
        prog='debug_cmd',
        description='Debug linux command error by using GPT/LLM.',
    )

    parser.add_argument('command', nargs='*', type=str, help='command and arguments separated by spaces. cannot use pipe(|), redirect(>), etc.')  # nopep8
    parser.add_argument('-c', type=str, help='shell string like "sh -c ..."')
    parser.add_argument('-f', action=argparse.BooleanOptionalAction, help='even if the return code is successful(0), the process is executed forcefully.')  # nopep8
    args = parser.parse_args()

    if not args.command and not args.c:
        parser.print_help()
        sys.exit(1)

    cmd = args.c if args.c else [arg for arg in args.command if arg]
    force = args.f

    return cmd, force


def _exec_command(cmd: Union[str, list[str]]) -> (int, str):
    """
    :param cmd: command line arguments.
    :return: (return code, stdout+stderr)
    """
    env = os.environ.copy()
    cwd = os.getcwd()

    # commands が list か str (-c option) かで、shell を経由するかどうか分けてる.
    proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) \
        if type(cmd) is list \
        else subprocess.Popen(cmd, shell=True, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def _print(text: str) -> str:
        sys.stdout.write(text)
        return text

    # print & concat.
    lines = ''.join([_print(line) for line in _get_stdout_by_lines(proc)])

    return proc.returncode, lines


def _get_stdout_by_lines(proc: subprocess.Popen) -> Generator[str, None, None]:
    """
    :param proc: sub process.
    :return: 標準出力 (行毎).
    """
    while True:
        line = proc.stdout.readline()

        if not line and proc.poll() is not None:
            break

        yield line.decode(_CLI_ENCODING)


def _ask_llm_about_error(cmd: str, return_code: int, error_message: str) -> str:
    """
    :param str cmd: command line arguments.
    :param int return_code: return code.
    :param str error_message: error message.
    :return: an answer.
    """
    # split error message.
    docs = _split_error_message(error_message)

    # create a prompt.
    query_text = _get_question(cmd, return_code)

    # using refine.
    question_prompt = _get_prompt_template(query_text)
    refine_prompt = _get_refine_template(query_text)
    chain = load_summarize_chain(_LLM, chain_type='refine', question_prompt=question_prompt, refine_prompt=refine_prompt)  # nopep8

    return chain.run(docs)


def _split_error_message(error_message: str) -> list[Document]:
    """
    Split error message for LangChain Refine.

    :param str error_message: error message.
    :return: a list of LangChain documents.
    """
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=_LLM.max_tokens / 1,
        chunk_overlap=_LLM.max_tokens / 10,
        length_function=len,
    )
    texts = text_splitter.split_text(error_message)
    documents = [Document(page_content=t) for t in texts]

    return documents


def _get_question(cmd: str, return_code: int) -> str:
    """
    :param cmd: command line arguments.
    :param return_code: return code.
    :return: llm prompt.
    """
    arch = platform.machine()
    cwd = os.getcwd()
    env_str = ','.join(os.environ.keys())

    return (
        'あなたは Mac, Windows, Unix, Linux 系 OS のターミナル上で発生したコマンドのエラーの解消を手助けする AI アシスタントです。'
        '質問者が使っているパソコンの情報は次の通りです。\n'
        f'エラーが発生したコマンド: {cmd}\n'
        f'コマンドの終了コード: {return_code}\n'
        f'OS名, OSバージョン: {_get_os()}\n'
        f'CPUアーキテクチャ: {arch}\n'
        f'カレントディレクトリ: {cwd}\n'
        f'環境変数: {env_str}\n'
        'これらの情報と与えられたエラーメッセージを元に、エラーの原因とその解決策を示して下さい: '
    )


def _get_prompt_template(query_str: str) -> PromptTemplate:
    """
    Create new prompt template in japanese.

    :param str query_str: query string.
    :return: A prompt template.
    :see: https://python.langchain.com/en/latest/modules/chains/index_examples/summarize.html#the-refine-chain
    """
    prompt_template = (
        "We have provided context information (error message) below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given this information, please answer the following question in Japanese.\n"
        "------------\n"
        f"{_escape_for_prompt_template(query_str)}\n"
        "------------\n"
    )

    return PromptTemplate(template=prompt_template, input_variables=["text"])


def _get_refine_template(query_str: str) -> PromptTemplate:
    """
    Create new refine template in japanese.

    :param str query_str: query string.
    :return: A refine template.
    :see: https://python.langchain.com/en/latest/modules/chains/index_examples/summarize.html#the-refine-chain
    """
    refine_template = (
        "Your job is to produce a final answer.\n"
        "We have provided an existing answer below up to a certain point.\n"
        "------------\n"
        "{existing_answer}\n"
        "------------\n"
        "We have also provided original question for existing answer below.\n"
        "------------\n"
        f"{_escape_for_prompt_template(query_str)}\n"
        "------------\n"
        "We have the opportunity to refine the existing answer"
        "(only if needed) with some more context (error message) below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original answer in Japanese.\n"
        "If the context isn't useful, return the original answer."
    )

    return PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )


def _escape_for_prompt_template(text: str) -> str:
    """
    :param text:
    :return:
    :see: https://github.com/hwchase17/langchain/issues/1660#issuecomment-1469320129
    """
    return text.replace('{', '{{').replace('}', '}}')


def _get_os() -> str:
    """
    :return: '{OS name} {OS version}'.
    """
    os_name = platform.system()
    os_version = platform.version()

    if os_name != 'Darwin':
        return f'{os_name} {os_version}'

    return f'macOS {platform.mac_ver()[0]}'


if __name__ == '__main__':
    main()

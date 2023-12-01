import os
import sys
import openai
import argparse
import tiktoken

from loguru import logger

from typing import Tuple

from rich.console import Console
from rich.markdown import Markdown
from colored import Fore, Back, Style

from trs.main import TRS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='trs-cli',
        description='Chat with and summarize CTI reports'
    )

    parser.add_argument(
        '-c', '--chat',
        required=True,
        action='store_true',
        help='Enter chat mode'
    )

    args = parser.parse_args()

    OPENAI_KEY = os.environ.get('OPENAI_API_KEY')
    if OPENAI_KEY is None:
        logger.error('OPENAI_API_KEY environment variable not set')
        sys.exit(1)

    trs = TRS(openai_key=OPENAI_KEY)

    COMMAND_HANDLERS = {
        '!summ': trs.summarize,
        '!detect': trs.detections,
        '!custom': trs.custom
    }

    if args.chat:
        console = Console()
        print(f'{Style.BOLD}{Fore.cyan_3}commands:{Style.reset}')
        print(f'* {Fore.cyan_3}!summ <url>{Style.reset} - summarize a threat report')
        print(f'* {Fore.cyan_3}!detect <url>{Style.reset} - identify detections in report')
        print(f'* {Fore.cyan_3}!custom <prompt_name> <url>{Style.reset} - process URL with a custom prompt')
        print(f'* {Fore.cyan_3}!exit|!quit{Style.reset} - exit application')

        print(f'{Style.BOLD}{Fore.dark_orange_3b}ready to chat!{Style.reset}\n')

        try:
            while True:
                prompt = input('💀 >> ').strip()
                
                if prompt.lower() in ['!exit', '!quit', '!q', '!x']:
                    logger.info('exiting')
                    break

                command, *args = prompt.split()
                handler = COMMAND_HANDLERS.get(command.lower())
                
                if handler:
                    result = handler(*args)

                    if command.lower() == '!summ':
                        summary, mindmap, iocs = result
                        print('🤖 >>')
                        console.print(Markdown(summary))
                        console.print(Markdown(mindmap))
                        print(iocs)

                    else:
                        print('🤖 >>')
                        console.print(Markdown(result))
                
                else:
                    result = trs.qna(prompt=prompt)
                    print('🤖 >>')
                    console.print(Markdown(result))


        except KeyboardInterrupt:
            logger.info('caught keyboard interrupt, exiting')
            sys.exit(0)

        except Exception as err:
            logger.error(f'error: {err}')
            sys.exit(1)


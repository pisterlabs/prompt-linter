import os
import sys
import traceback
import argparse
import json
from prompt_toolkit import print_formatted_text, HTML
from prompt_toolkit import prompt
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.callbacks import get_openai_callback

import vectordb
import settings


class BaseFAQBot(object):
    def __init__(self):
        self._db = None
        self._debug = False
        self._chain = None
        k = os.getenv("OPENAI_API_KEY")
        if not k:
            raise Exception("OPENAI_API_KEY not set")
        self.get_db()

    def set_debug(self, flag):
        if flag is True or flag in ['1', 'true', 'True', 'TRUE']:
            self._debug = True
            return True
        elif flag is False or flag in ['0', 'false', 'False', 'FALSE']:
            self._debug = False
            return False
        return None

    def is_debug_enabled(self):
        return self._debug

    def get_cost(self):
        return self._set_cost

    def get_db(self):
        if self._db is None:
            self._db = vectordb.Loader.load(settings.VECTOR_DATABASE)
        return self._db

    def _get_llm_chain(self):
        if self._chain is None:
            system_template = settings.FAQBOT_SYSTEM_TEMPLATE
            messages = [
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
            chat_prompt = ChatPromptTemplate.from_messages(messages)
            chain_type_kwargs = {"prompt": chat_prompt}
            llm = ChatOpenAI(model_name=settings.FAQBOT_OPENAI_MODEL, 
                             temperature=settings.FAQBOT_OPENAI_TEMPERATURE, 
                             max_tokens=settings.FAQBOT_OPENAI_MAX_TOKENS,
                             request_timeout=settings.FAQBOT_OPENAI_REQUEST_TIMEOUT)
            self._chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.get_db().as_retriever(),
                return_source_documents=True,
                chain_type_kwargs=chain_type_kwargs
            )
        return self._chain

    def parse_question(self, question):
        q = []
        for line in question.splitlines():
            for l in line.split('. '):
                q.append(l.strip()+'.\n')
        return '- '.join(q)

    @classmethod
    def perror(cls, error):
        print_formatted_text(HTML('<p fg="ansired">ERROR: {}</p>'.format(error)))
        print('\n')

    def ask(self, question):
        if not question:
            return json.dumps({"status": "error",
                               "error": "Question is required"})
        data = self.query_as_dict(question)
        return json.dumps({"status": "success",
                           'response': data})

    def _query(self, question):
        question = self.parse_question(question)
        result = {}
        if self.is_debug_enabled():
            with get_openai_callback() as cb:
                result = self._get_llm_chain()(question)
                result["stats"] = {'total_tokens': cb.total_tokens,
                            'prompt_tokens': cb.prompt_tokens,
                            'completion_tokens': cb.completion_tokens,
                            'successful_requests': cb.successful_requests,
                            'total_cost': cb.total_cost}
        else:
            result = self._get_llm_chain()(question)
        return result

    def query_as_dict(self, question):
        result = self._query(question)
        data_sources = list(set([doc.metadata['source'] for doc in result['source_documents']]))
        response = {'question': question,
                    'answer': result['answer'], 
                    'sources': data_sources,
                    }
        if self.is_debug_enabled():
            response['stats'] = result['stats']
            response['raw_response'] = str(result)
        return response

    def query_as_text(self, question):
        result = self._query(question)
        data_sources = set(['- '+doc.metadata['source'] for doc in result['source_documents']])
        data_sources = '\n'.join(tuple(data_sources))
        output_text = f"""
# Question
{question}

# Answer
{result['answer']}

# Sources 
{data_sources}

"""
        if self.is_debug_enabled():
            msg = f'\n\n# Cost\n{result["stats"]}\n'
            msg += f'\n\n# Raw response\n{result}\n'
            output_text += msg

        return output_text

    def query_and_print_result(self, question):
        print(self.query_as_text(question))




class FAQBot(BaseFAQBot):
    '''
    FAQBot is a chatbot that answers programming questions.

    # Programming mode
    bot = FAQBot()
    bot.set_debug(True)
    result = bot.ask(question="send an SMS")

    # Prompt mode
    bot = FAQBot()
    bot.run()

    # CLI mode
    FAQBot.cli()
    '''
    def __init__(self, banner='FAQBot'):
        self._banner = f'<p fg="ansiwhite">{banner}</p><p fg="ansired"> ("/help" for help)</p>'
        super().__init__()
        self._set_cost = True

    @classmethod
    def cli(cls, banner='FAQBot'):
        parser = argparse.ArgumentParser(description="FAQBot")
        parser.add_argument("-a", "--ask", type=str, default="", help="Question to ask (required in CLI mode). Use '-' for stdin.")
        parser.add_argument("-m", "--mode", type=str, choices=['prompt', 'cli'], default="prompt", help="CLI mode or Prompt mode - default: prompt")
        parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode (show OpenAI API cost and response)")
        parser.add_argument("-b", "--banner", type=str, default="FAQBot", help=f"Banner text (only used in prompt mode) - default {banner}")
        args = parser.parse_args()
        debug = args.debug
        banner = args.banner
        mode = args.mode
        ask = args.ask
        if ask == '-':
            ask = sys.stdin.read()
        if ask and mode == 'prompt':
            cls.perror("-a/--ask cannot be use in Prompt mode")
            parser.print_help()
            sys.exit(1)
        if mode == 'cli' and not ask:
            cls.perror("-a/--ask is required in CLI mode")
            parser.print_help()
            sys.exit(1)
        bot = cls(banner=banner)
        debug is True and bot.set_debug(True)
        if mode == 'prompt':
            bot.run()
            return
        elif mode == 'cli':
            bot.cli_run(ask)
            return

        parser.print_help()
        sys.exit(1)

    def cli_run(self, question):
        self.query_and_print_result(question)
        sys.exit(0)

    def run(self):
        print_formatted_text(HTML(self._banner))
        while True:
            try:
                self._wait_for_input()
            except (KeyboardInterrupt, EOFError):
                print("Bye!")
                sys.exit(0)
            except Exception as e:
                self.perror(f"Oops: {e}")
                traceback.print_exc()
                continue

    def _cmd_exit(self, msg='Bye!'):
        print(msg)
        sys.exit(0)

    def _cmd_clear(self):
        print("\033c")

    def _cmd_banner(self):
        print_formatted_text(HTML(self._banner))
        print('\n')

    def _cmd_ask(self, query):
        query = query.replace("/ask ", "")
        if not query:
            self.perror("You must enter a question to ask")
            return
        self.query_and_print_result(query)
        print('\n')

    def _cmd_help(self):
        print("\tType '/ask [QUESTION]' to ask a question")
        print("\tType '/banner' show banner")
        print("\tType '/clear' to clear the screen")
        print("\tType '/debug' show if debug mode is enabled or disabled")
        print("\tType '/debug [true|false]' debug info including cost")
        print("\tType '/quit' to exit")
        print('\n')

    def _cmd_debug(self):
        if self.is_debug_enabled():
            print("Debug enabled")
        else:
            print("Debug disabled")
        print('\n')

    def _cmd_debug_change(self, debug):
        debug = debug.strip()
        res = self.set_debug(debug)
        if res is True:
            print(f"Debug enabled")
        elif res is False:
            print(f"Debug disabled")
        else:
            self.perror("Please specify true or false")
        print('\n')

    def _wait_for_input(self):
        query = prompt(">>> ")
        if query == "/quit":
            self._cmd_exit('Bye!')
        elif query == "/banner":
            self._cmd_banner()
        elif query == "/clear":
            self._cmd_clear()
        elif query.startswith("/ask"):
            self._cmd_ask(query)
        elif query == "/help":
            self._cmd_help()
        elif query == "/debug":
            self._cmd_debug()
        elif query.startswith("/debug"):
            try:
                _cmd, _debug = query.split(" ", 1)
            except ValueError:
                self.perror("Please specify true or false")
                return
            self._cmd_debug_change(_debug)
        else:
            if query.strip() == "":
                return
            self.perror("Unknown command, type '/help' for help")


if __name__ == "__main__":
    FAQBot.cli()



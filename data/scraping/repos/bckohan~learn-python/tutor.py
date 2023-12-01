import typer
import re
from pathlib import Path
import os
from contextlib import contextmanager
import subprocess
from functools import cached_property
from enum import Enum
from termcolor import colored
from typing import Optional, Union, List
from types import FunctionType
from uuid import uuid1, UUID
from learn_python.tests.tasks import Task, TaskStatus
from learn_python.tests.tests import tasks
from learn_python.client import CourseClient
from learn_python.register import LLMBackends
from learn_python.doc import (
    task_map,
    TaskMapper,
    clean as clean_docs,
    build as build_docs
)
from learn_python.register import Config
import gzip
import json
from datetime import datetime
from dateutil.tz import tzlocal
from glob import glob
from warnings import warn
import asyncio
import sys
from rich.console import Console
from rich.markdown import Markdown
from learn_python.utils import (
    ConeOfSilence,
    GzipFileHandler,
    localize_identifier,
    ROOT_DIR,
    strip_colors,
    lp_logger,
    DateTimeEncoder,
    configure_logging,
    formatter
)
from learn_python import main
import shutil
import logging

# don't remove this - adds additional editing features to input()
import readline


LOG_DIR = ROOT_DIR / 'learn_python/delphi/logs'


def now():
    """Get the current time in the system's current timezone"""
    return datetime.now(tz=tzlocal())



class TerminateSession(Exception):
    """Thrown if the tutor determines the session should end"""


class RestartSession(Exception):
    """Thrown if the tutor determines the session should be restarted - possibly for a new task"""


class ConfigurationError(Exception):
    """Thrown if there's something wrong with the setup of the Tutor such that it can't start properly"""


class RePrompt(Exception):
    """
    Thrown when we need to break the response cycle and get the AI to respond to a different message,
    the string representation of the exception will be used as the new prompt. The session will not be
    broken.
    """


class Tutor:
    """
    A base class for Tutor implementations. This class is responsible for the common tutoring tasks
    like defining the tutor's personality and fetching structured context regarding the gateway
    assignments and lessons. Implementations of tutors for specific LLMs or AI platforms should inherit 
    from this class.

    todo:
        2) function for selecting task to get help with
        4) log to web server

    The base class will attempt to render all tutor text to the terminal as markdown.

    For a backend to fully work it needs to also provide a mechanism to make function calls and
    pass arguments to those functions, though this is not strictly required as users can manually
    indicate which tasks they need help with and ctrl-D out of the session.
    """

    # each session is given a unique identifier
    engagement_id: UUID = None
    session_id: int = -1

    log: dict = None

    # the session message chain
    messages: List[str] = None
    resp_json: List[str] = None  # optional extra backend response data
    engagement_start: datetime = None
    session_start: datetime = None
    session_end: datetime = None

    # if our session pertains to a particular task, they will be here
    task_test: Task = None
    task_docs: TaskMapper.AssignmentDocs = None

    LOG_RGX = re.compile('delphi_(?P<id>[\w-]+)[.]json(?:[.](?P<ext>gz))?$')

    # accept tasks specified as their singular names, their pytest identifiers or the pytest function name
    TASK_NAME_RGX = re.compile('^((?:test_gateway[\d]*_)|(.*[:]{2}))?(?P<task_name>[\w]+)$')

    logger = logging.getLogger('delphi')
    file_handler: GzipFileHandler

    BACKEND: LLMBackends

    # some guards against recursive loops
    last_function: Optional[FunctionType] = None
    last_function_kwargs: Optional[dict] = None
    no_prompt_rounds: int = 0
    NO_PROMPT_ROUNDS_LIMIT: int = 4

    API_KEY_FILE = None

    INITIAL_NOTICE = """
---
Hi, I'm **Delphi**! \U0001F44B

- To **terminate the session**, you can:
  - Type `ctrl-D`
  - Or kindly tell me you're finished.

- Need **help with an assignment** or curious about **Python and programming**? Just ask!

- I can also **build the docs** for you and **re-evaluate assignments** after you've made changes. Let's get started!
---
    """

    closed: bool = True

    def __init__(self, api_key=None):
        self.engagement_id = uuid1()
        self.resp_json = []
        if not LOG_DIR.is_dir():
            os.makedirs(LOG_DIR, exist_ok=True)
        self.file_handler = GzipFileHandler(str(LOG_DIR / f'delphi_{self.engagement_id}.log'))

        # setup delphi logging to a unique file for each engagement
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)
        self.logger.setLevel(logging.DEBUG)
        self.file_handler.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.no_prompt_rounds = 0
        self.log = {}

        if api_key is None and self.API_KEY_FILE and self.API_KEY_FILE.is_file():
            self.api_key = self.API_KEY_FILE.read_text().strip()
        if not self.api_key and self.API_KEY_FILE is not None:
            with open(self.API_KEY_FILE, 'a'):
                os.utime(self.API_KEY_FILE, None)


    def input(self, prompt):
        """
        A wrapper around input() to allow for mocking during testing.
        """
        return input(prompt)
    
    @classmethod
    def write_key(cls, key=None):
        if cls.API_KEY_FILE is None and key is not None:
            raise NotImplementedError(
                f'Tutor backend {cls.BACKEND} does not support write_key().'
            )
        with open(cls.API_KEY_FILE, 'w') as key_f:
            key_f.write(key if isinstance(key, str) else key.encode('utf-8'))

    @property
    def me(self):
        return 'Delphi'

    @property
    def directive(self):
        """Who is the tutor?"""
        return (
            f'Your name is {self.me}. My name is {Config().student}. You are a friendly and encouraging tutor who '
            f'will help me learn the Python programming language. You will not write any code for me even '
            f'if I ask, instead you will use natural language to explain any errors and suggest avenues '
            f'of approach. Please address me by my name.'
        )

    def is_help_needed(self, task: Optional[Union[Task, str]] = None):
        """Check if the student needs help, if they do launch the tutor, if not return"""
        use_tutor = self.input(f'Would you like assistance from {self.me}? (y/n): ')
        for word in use_tutor.lower().split():
            if word in ['yes', 'y', 'help', 'please', 'thank', 'thanks', 'thankyou', 'ok', 'affirmative']:
                with delphi_context():
                    self.init(task=task).submit_logs()
                return
    
    def prompt(self, msg: Optional[str] = None, role='user'):
        """
        The prompt/response cycle. If a msg is passed in, the user will
        not be given a chance to input text and the message will be immediately
        sent to the tutor backend.
        """
        self.logger.info('prompt()')
        if msg is None:
            self.no_prompt_rounds = 0
            msg = self.input('> ')
        else:
            self.no_prompt_rounds += 1

        if self.no_prompt_rounds > self.NO_PROMPT_ROUNDS_LIMIT:
            # we might be caught in some kind of loop - must guard against this!
            raise TerminateSession(
                f'{self.me} has gotten confused and exceeded the number of rounds without user input.'
            )
        
        Console().print(  # render as markdown to the terminal.
            Markdown(
                self.push(  # log the response message
                    'assistant',
                    self.handle_response(  # call any functions
                        asyncio.run(self.get_response(msg, role=role))  # asynchronously send the message
                    )
                ) or ''  # response may have been null if certain functions were called
            )
        )

    async def spinner(self):
        """
        Asynchronously display a spinner on the terminal. The responses from LLMs can take
        seconds, so its important to give the user a visual cue that something is happening
        and they should wait.
        """
        symbols = ['|', '/', '-', '\\']
        while True:
            for symbol in symbols:
                sys.stdout.write(f'\r{symbol}')
                sys.stdout.flush()
                await asyncio.sleep(0.1)

    async def get_response(self, message='', role='user'):
        """
        Get the response to the message from the user. If no message, rely on whats
        on the messages stack. This will display a progress spinner on the terminal
        until send completes and a response is available.
        """
        self.logger.info(f'get_response()')
        if message:
            self.push(role, message)
        spinner_task = asyncio.create_task(self.spinner())
        response = await self.send()
        spinner_task.cancel()
        sys.stdout.write('\r')
        sys.stdout.flush()
        return response
    
    async def send(self):
        """
        Send the message chain and return a response.
        """
        raise NotImplementedError(
            f'Extending LLM backends must implement send.'
        )
    
    def handle_response(self, response):
        """
        Process the response object and return the message from it. If the response asks for any functions
        to be called, they should be executed here.

        :return: A 2-tuple where the first element is the message and the second is a boolean indicating
            set to true if this response called a function and false if it did not.
        """
        raise NotImplementedError(
            f'Extending LLM backends must implement handle_response.'
        )
    
    def get_task_description(
        self,
        test,
        docs,
        hints=False,
        requirements=False,
        code=False
    ):
        """
        Get a natural language description for the task.

        :param test: The Task object holding information regarding the test.
        :param docs: The AssignmentDocs object holding information about the task assignment.
        :param hints: If true, include the hints in the description (default: false)
        :param requirements: If true, include the requirements in the description (default: false)
        :param code: If true, include the current task implementation code in the description (default: false)
        :return: A string containing the natural language description of the task.
        """
        test.run()
        description = f'The task {docs.name} asks me to:\n{docs.todo}\n'
        if self.task_test.status == TaskStatus.PASSED:
            description += f'The test for {docs.name} is passing!\n'
        elif self.task_test.status in [TaskStatus.ERROR, TaskStatus.FAILED]:
            description += f'The test for {docs.name} is failing with this error: {test.error_msg}\n'
        elif self.task_test.status == TaskStatus.SKIPPED:
            description += f'I have not attempted to implement {docs.name} yet.\n'
        if requirements and docs.requirements:
            reqs = "\n".join([f'* {req}' for req in docs.requirements])
            description += f'It has the following requirements:\n{reqs}\n'
        if hints and docs.hints:
            hnts = "\n".join([ f'* {hint}' for hint in docs.hints])
            description += f'I have been given the following hints: \n{hnts}\n'
        if code and test.implementation:
            description += f'My current implementation of {docs.name} looks like:\n```python\n{test.implementation}```'
        return description

    def init_for_task(self):
        """Reinitialize AI tutor context for help with the set tasks"""
        # sanity check
        assert self.task_test and self.task_docs, f'Cannot initialize {self.me} for a task without a test and documentation to rely on.'
        self.logger.info('init_for_task(%s)', self.task_test.name)
        self.task_test.run()
        message = f'I have set the task I need help with to "{self.task_test.name}". '
        message += self.get_task_description(
            self.task_test,
            self.task_docs,
            hints=True,
            requirements=True,
            code=True
        )

        dependency_set = set()
        def get_all_dependencies(test_docs):
            for module_dep, task_dep in test_docs.dependencies:
                module, test, docs = self.get_task_info(task_dep, module_dep)
                if module and test and docs:
                    dependency_set.add((module, test, docs))
                    get_all_dependencies(docs)
        
        get_all_dependencies(self.task_docs)

        for module, test, docs in dependency_set:
            if module and test and docs:
                message += self.get_task_description(test, docs, code=True)

        self.push('system', message.strip())

    def push(self, role, message='', function_call=False):
        """Add message to history"""
        self.logger.info('push(role=%s, message=%s)', role, message)
        if message:
            self.messages.append({
                'role': role,
                'content': message,
                'timestamp': now(),
                'function_call': function_call,
                'backend_extra': self.pop_resp_json()
            })
        return message
    
    def pop_resp_json(self):
        if self.resp_json:
            self.resp_json.pop()
        return {}
    
    def possible_tasks(self, task_name: str, module: Optional[str] = None):
        """
        Get a list of possible tasks based on the task_name. The list will be
        a 2-tuple where the first element is the module name string and the second
        element is the Task instance.

        :param task_name" The name of the task
        """
        possibles = []
        mtch = self.TASK_NAME_RGX.match(task_name)
        to_search = {module: tasks.get(module, {})} if module else tasks
        if mtch:
            task_name = mtch.groupdict()['task_name']
            possibles = []
            for module, mod_tasks in to_search.items():
                if task_name in mod_tasks:
                    possibles.append((module, mod_tasks[task_name]))
        self.logger.info('possible_tasks(%s) = %s', task_name, possibles)
        return possibles
    
    def try_select_task(self, task_name):
        """
        Given a task_name which may or may not match a real task, try to select the task to get
        help with. We use a hybrid AI/choices approach. A complicating factor is that different
        modules may have tasks of the same name.

        :param task_name: The name of the task 
        """
        self.logger.info('try_select_task(%s)', task_name)
        while not (possibles := self.possible_tasks(task_name)):
            qry = f'{task_name} is not an assignment, the assignments are:\n'
            not_working = []
            for module, mod_tasks in tasks.items():
                qry += f'In {module}: {",".join(mod_tasks.keys())}\n'
                for task_name, task in mod_tasks.items():
                    task.run()
                    if task.status in [TaskStatus.ERROR, TaskStatus.FAILED]:
                        not_working.append((module, task_name)) 

            for broken in not_working:
                qry += '\n'.join(f'{broken[1]} in {broken[0]} is not working.')
            
            qry += '\nSet which task you think the user might be referring to.'
            self.prompt(qry, role='system')
            task_name = self.input('? ')

        if len(possibles) == 1:
            self.task_test = possibles[0][1]
        else:
            while self.task_test is None:
                prompt = f'Which task do you want help with?:\n'
                for idx, candidate in enumerate(possibles):
                    prompt += f'[{idx}] {candidate[0]}: {candidate[1].name}'
                try:
                    selection = self.input(prompt)
                    self.task_test = possibles[int(selection)][1]
                except (TypeError, ValueError):
                    self.try_select_task(selection)
                except IndexError:
                    print(colored(f'{selection} is not a valid task number.', 'red'))

    def init(
        self,
        task: Optional[Union[Task, str]] = None,
        notice: Optional[str] = INITIAL_NOTICE
    ):
        """
        Initialize a tutor engagement. This may contain multiple sessions. See start_session()
        """
        self.engagement_start = self.engagement_start or now()
        if not Config().is_registered():
            Config().register()
        os.makedirs(LOG_DIR, exist_ok=True)
        self.logger.info('%s.init(%s, %s)', self.__class__.__name__, task, strip_colors(notice))
        try:
            self.start_session(task=task, notice=notice)
        except RestartSession as err:
            # a little tail recursion never hurt anybody
            self.init(task=self.task_test, notice=str(err))
        self.close_session()
        if self.file_handler:
            self.file_handler.close()
        return self
    
    def get_task_info(self, task_name: str, module: Optional[str] = None):
        module, test = self.get_test(task_name, module=module)
        if test:
            return module, test, self.get_docs(module, task_name)
        return None, None, None

    def get_test(self, task_name: str, module: Optional[str] = None):
        possibles = self.possible_tasks(task_name, module=module)
        if len(possibles) == 1:
            return possibles[0][0], possibles[0][1]
        return None, None

    def get_docs(self, module: str, task_name: str):
        with ConeOfSilence():  # this might trigger a doc parse which is very chatty!
            return task_map().get_task_doc(module, task_name)
    
    def close_session(self):
        if self.session_id >= 0 and not self.closed:
            self.session_end = now()
            self.log_session()
            self.closed = True

    def start_session(
        self,
        task: Optional[Union[Task, str]] = None,
        notice: str = '',
        initial_prompt: str = ''
    ):
        """
        A session can be thought of as a conversation with the LLM. There are tight token (word)
        limits on how long conversations can be with different LLM backends so during the course
        of chats it may be necessary or make sense to restart the conversation thread. We use
        a constant engagement identifier to group sessions together as part of a larger interaction.

        Start a new tutor session. This may be part of a previous engagement chain, if so
        the engagement ID will not increment.

        As each session ends it will be logged to disk and the server collector if configured to
        do so.

        :param task: The Task instance or name of the task to get help with. May be null
        :param notice: An initial notice or prompt to display to the user
        """
        self.close_session()
        self.session_id += 1
        self.closed = False
        self.messages = []
        self.session_start = now()
        self.session_end = None
        self.task_test = None
        self.task_docs = None

        self.logger.info('start_session(%s) = %d', task, self.session_id)
        
        if notice:
            Console().print(Markdown(notice))
        try:
            if isinstance(task, Task):
                self.task_test = task
            elif task and isinstance(task, str):
                self.try_select_task(task)

            if self.task_test:
                self.task_docs = self.get_docs(self.task_test.module, self.task_test.name)
                if self.task_docs:
                    self.init_for_task()
                else:
                    # this shouldn't be possible if doc check is passing,
                    # unless students delete docs
                    warn(
                        f'Unable to resolve instructions for task: '
                        f'[{self.task_test.module}] {self.task_test.name}, {self.me} '
                        f'will not have the needed context.'
                    )

            if initial_prompt:
                self.push(role='system', message=initial_prompt)
            first = True
            while True:
                self.prompt('' if first and self.messages else None)
                first = False

        except TerminateSession as term:
            print(colored(str(term) or 'Goodbye.', 'blue'))
        except (TerminateSession, EOFError, KeyboardInterrupt):
            print(colored(f'Goodbye.', 'blue'))
        except RePrompt as new_prompt:
            return self.start_session(task=self.task_test, initial_prompt=str(new_prompt))

        self.close_session()

    def terminate(self):
        """AI invocable function to terminate the engagement."""
        self.logger.info('terminate()')
        raise TerminateSession()
    
    def test(self):
        """
        AI invocable function. Re-executes the current task if there is one. If the task is
        passing the session will be terminated.
        """
        if not self.task_test:
            self.logger.info('test(None)')
            return
        
        self.logger.info('test(%s)', localize_identifier(self.task_test.identifier))
        print(colored(f'poetry run pytest {localize_identifier(self.task_test.identifier)}', 'blue'))

        self.task_test.run(force=True)
        if self.task_test.status == TaskStatus.PASSED:
            print(colored(
                f'{self.task_test.name} is passing now! Good job! '
                f'Do not hesitate to ask me for help again!',
                'green'
            ))
            raise TerminateSession()
        if self.task_test.error:
            print(colored(self.task_test.error_msg, 'red'))
        raise RestartSession()
    
    def docs(self, clean=True):
        """
        AI invocable function to build or clean the documentation.
        """
        self.logger.info('docs(clean=%s)', clean)
        if clean:
            print(colored('poetry run doc clean', 'blue'))
            clean_docs()
            return
        print(colored('poetry run doc build', 'blue'))
        build_docs()

    def set_task(self, task_name: str):
        """
        AI invocable function to set the task for which the student needs help.
        The session will be restarted with the task in question.
        """
        self.logger.info('set_task(%s)', task_name)
        if self.task_test and self.task_test.name == task_name:
            # Guard against the AI mistakenly trying to reset the task - if this happens
            # we need to tell it what to do.
            raise RePrompt(
                f'No, the {task_name} task is already set. I am asking for help about how to approach solving it!'
            )
        self.task_test = task_name
        raise RestartSession()

    def log_session(self):
        if not self.log:
            self.log = {
                'engagement_id': str(self.engagement_id),
                'sessions': [],
                'timestamp': self.engagement_start.isoformat(),
                'stop': None,
                'tz_name': now().strftime('%Z'),
                'tz_offset': now().utcoffset().total_seconds() // 3600,
                'backend': self.BACKEND.value,
                'backend_log': {},
                'log_path': self.file_handler.baseFilename
            }
        
        task = {}
        if isinstance(self.task_test, Task) and self.task_test.module_number:
            task = {
                'module': self.task_test.module_number,
                'identifier': localize_identifier(self.task_test.identifier)
            }

        self.log['sessions'].append({
            'session_id': self.session_id,
            'timestamp': self.session_start.isoformat(),
            'stop': self.session_end.isoformat(),
            'exchanges': self.messages
        })
        if task:
            self.log['sessions'][-1]['assignment'] = task
        try:
            with gzip.open(
                LOG_DIR / f'delphi_{self.engagement_id}.json.gz',
                'wt',
                encoding='utf-8'
            ) as f:
                self.log['stop'] = now().isoformat()
                self.log['backend_log'] = self.backend_log
                json.dump(self.log, f, cls=DateTimeEncoder, indent=4)
        except Exception as err:
            # this logging is not crucial - lets not confuse students if any errors pop up
            self.logger.exception(f'Exception encountered logging the session.')

    @staticmethod
    def submit_logs():
        """
        Submit logs to the lesson server - this will also delete them if submission
        is successful
        """
        logs = [
            *glob(str(LOG_DIR / 'delphi_*.json')),
            *glob(str(LOG_DIR / 'delphi_*.json.gz'))
        ]
        log_by_id = {}
        for log in logs:
            mtch = Tutor.LOG_RGX.search(log)
            if mtch:
                log_by_id[mtch.groupdict()['id']] = {
                    'record': LOG_DIR / log,
                    'ext': mtch.groupdict().get('ext')
                }
        
        submit_count = 0
        errors = 0
        for eng_id, log in log_by_id.items():
            try:
                engagement_data = {}

                # be robust to students decompressing logs
                if log['ext'] and log['ext'].lower() == 'gz':
                    with gzip.open(log['record'], 'rt', encoding='utf-8') as f:
                        engagement_data = json.load(f)
                else:
                    with open(log['record'], 'rt', encoding='utf-8') as f:
                        engagement_data = json.load(f)

                log_path = engagement_data.pop('log_path', None)
                if log_path:
                    # be robust to students decompressing logs
                    for file in [
                        *glob(f'{log_path}.gz'),  # prioritize the gzipped log
                        *glob(log_path),
                        *glob(f'{log_path}*')
                    ]:
                        if os.path.exists(file):
                            log_path = Path(file)
                            break

                if not log_path.is_file():
                    log_path = None

                if engagement_data:
                    CourseClient().post_engagement(engagement_data)
                    if log_path:
                        CourseClient().post_log(log_path)
                        os.remove(log_path)
                    os.remove(log['record'])
                    submit_count += 1
            except Exception:
                lp_logger.exception(f'Exception encountered submitting log {log["record"]} to server.')
                errors += 1

        return submit_count, errors
    
    @property
    def function_map(self):
        """
        A map of function names to function callables. Useful for turning an AI directed
        call into an actual function call
        """
        return {
            'terminate': self.terminate,
            'test': self.test,
            'docs': self.docs,
            'set_task': self.set_task
        }
    
    def call_function(self, function_name, **kwargs):
        """
        Attempt to invoke the function of the given name with the given arguments.
        """
        if not function_name:
            return
        def no_op(**kwargs):
            return None
        
        def kwarg_str():
            pairs = []
            for key, value in kwargs.items():
                pairs.append(f'{key}={value}')
            return ', '.join(pairs)
        
        func = self.function_map.get(function_name, no_op)
        if func == no_op:
            self.logger.info('unrecognized call %s == no_op()', function_name)
        else:
            self.logger.info('call %s', function_name)
        try:
            self.push('assistant', f'{function_name}({kwarg_str()})', function_call=True)
            return func(**kwargs)
        except RePrompt:
            # give our function a chance to salvage the session!
            func = None
            raise
        finally:
            if func != no_op and func == self.last_function and kwargs == self.last_function_kwargs:
                # guard against a recursive function loop!
                raise TerminateSession(
                    f'{self.me} is confused and has called {function_name}({kwarg_str()}) twice in row!'
                )
            self.last_function = func
            self.last_function_kwargs = kwargs

    @property
    def functions(self):
        """
        Data structure that describes the functions the AI should know how to call. This is in the
        OpenAI format.
        """
        funcs = [{
            'name': 'terminate',
            'description': 'Terminate the tutoring session.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': []
            },
        }, {
            'name': 'set_task',
            'description': 'Set the task the user wants help with.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'task_name': {
                        'type': 'string',
                        'description': 'The name of the task the user is asking for help with.',
                    },
                },
                'required': ['task_name']
            }
        }, {
            'name': 'docs',
            'description': 'Build the documentation, optionally just clean it.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'clean': {
                        'type': 'boolean',
                        'description': 'Do not build the docs, just clean them',
                    }
                }
            }
        }]
        if self.task_test:
            funcs.append({
                'name': 'test',
                'description': 'Check the student\'s solution by executing the test for '
                                'the task they are working on.',
                'parameters': {
                    'type': 'object',
                    'properties': {},
                    'required': []
                }
            }
        )
        return funcs

    @property
    def backend_log(self):
        """Any json serializable structure that a backend would like to be part of the log."""
        return None
    

@contextmanager
def delphi_context():
    """
    A context manager for a Delphi engagement - sets the current working directory
    to to the root of the repository. Probably unnecessary.
    """
    assert ROOT_DIR.is_dir()
    start_dir = os.getcwd()
    os.chdir(ROOT_DIR)
    yield
    os.chdir(start_dir)


_tutor = None
_explicitly_invoked = False
def tutor(llm = LLMBackends.OPEN_AI):
    """
    Get the active tutor, if this returns None then tutoring is not enabled.
    """
    global _tutor
    if _tutor and _tutor.backend is llm:
        return _tutor
    
    def try_instantiate():
        if llm is LLMBackends.OPEN_AI:
            from learn_python.delphi.openai import OpenAITutor
            return OpenAITutor()
        elif llm is LLMBackends.TEST:
            from learn_python.delphi.test import TestAITutor
            return TestAITutor()
        else:
            raise NotImplementedError(f'{llm} tutor backend is not implemented!')
    try:
        _tutor = try_instantiate()
    except ConfigurationError:
        if Config().try_authorize_tutor():
            _tutor = try_instantiate()
        elif _explicitly_invoked:
            raise
    except Exception:
        if _explicitly_invoked:
            raise
    return _tutor

@main(catch=True)
def delphi(
    task: Optional[str] = typer.Argument(
        None,
        help="The gateway task you need help with."
    ),
    submit_logs: Optional[bool] = typer.Option(
        False,
        '--submit-logs',
        help="Submit logs to the lesson server."
    ),
    clean_logs: Optional[bool] = typer.Option(
        False,
        '--clean-logs',
        help="Delete the logs directory."
    ),
    llm: Optional[LLMBackends] = Config().tutor.value
):
    """I need some help! Wake Delphi up!"""
    global _explicitly_invoked
    _explicitly_invoked = True
    from learn_python.register import do_report, lock_reporting
    lock_reporting()
    configure_logging()
    logging.getLogger('testing').info('[START] tutor')
    try:
        with delphi_context():
            try:
                if submit_logs:
                    submitted, errors = Tutor.submit_logs()
                    if submitted:
                        print(colored(f'Submitted {submitted} logs to the lesson server.', 'green'))
                    if errors:
                        print(colored(f'Encountered {errors} errors submitting logs to the lesson server.', 'red'))
                    return
                if clean_logs:
                    if LOG_DIR.is_dir():
                        shutil.rmtree(LOG_DIR)
                    return
                tutor(llm).init(task).submit_logs()
            except ConfigurationError as err:
                print(colored(str(err), 'red'))
    finally:
        logging.getLogger('testing').info('[STOP] tutor')
        lock_reporting(False)
        do_report()

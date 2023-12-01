import datetime
import re
from dataclasses import dataclass

import openai as openai
import pexpect.fdpexpect

from settings import settings
from utils import splited_print


@dataclass
class Response:
    final_prompt: str
    answer: str
    iterations: int
    total_tokens: int
    execution_total_cost: float


ansi_escape = re.compile(r'''
    \x1B  # ESC
    (?:   # 7-bit C1 Fe (except CSI)
        [@-Z\\-_]
    |     # or [ for CSI, followed by a control sequence
        \[
        [0-?]*  # Parameter bytes
        [ -/]*  # Intermediate bytes
        [@-~]   # Final byte
    )
''', re.VERBOSE)


def find_max_input_number(prompt):
    """
    Parse all numbers in format In [${number}] and return the max number
    """
    numbers = [int(x) for x in re.findall(r"In \[(\d+)\]", prompt)]
    return max(numbers) if numbers else 0


def request_prompt(prompt):
    return openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=90,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n\n", "Out[", "Collecting", "Requirement alread"]
    )


def sanitize_command(value):
    value = value.replace(
        '!pip install',
        '%pip --quiet --disable-pip-version-check  --no-python-version-warning install --root-user-action=ignore'
    )
    value = value.replace(
        '%pip install',
        '%pip --quiet --disable-pip-version-check  --no-python-version-warning install --root-user-action=ignore'
    )
    value = value.strip()
    value = value.lstrip('\r')
    value = value.rstrip('\r')
    value = value.lstrip('\n')
    value = value.rstrip('\n')
    return value


def sanitize_stdout(value):
    return ansi_escape.sub('', value.decode('utf-8')).rstrip('\n')


def sanitize_gpt3_response(value):
    value = value['choices'][0]['text']
    value = value.lstrip('\n')
    value = value.rstrip('\n')
    value = value.lstrip(' ')
    return value


def calc_execution_costs(total_tokens, cost_per_1000_tokens):
    return cost_per_1000_tokens * (total_tokens / 1000)


def process_question(
    question,
    max_iterations=5,
    apikey=None,
    cost_per_1000_tokens=None
) -> Response:
    openai.api_key = apikey or settings.OPENAI_API_KEY
    cost_per_1000_tokens = cost_per_1000_tokens or settings.MODEL_COST_DOLLARS

    total_tokens = 0

    now = datetime.datetime.now()
    today = now.strftime('%d %B %Y %A %H:%M:%S')
    prompt = settings.PROMPT.replace('{{today}}', today)
    prompt = prompt.replace('{{question}}', question)

    # open subprocess to run ipython

    # Create ipython process
    c = pexpect.spawn('ipython --colors=NoColor --nosep --simple-prompt')
    execution_number = 1
    c.expect_exact(f'In [{execution_number}]:')

    # got_answer indicates what response contains "Answer:"
    iteration = 0
    while True:
        if iteration > max_iterations:
            prompt += f"""
            ```
            """
        print('GPT-3 REQUEST')
        response = request_prompt(prompt)
        total_tokens += response['usage']['total_tokens']
        response = sanitize_gpt3_response(response)
        # If response contains "Answer:", then we are done
        if 'Answer:' in response:
            answer = response.split('Answer:')[1]
            prompt += response
            splited_print(prompt)
            return Response(
                final_prompt=prompt,
                answer=answer,
                iterations=iteration,
                execution_total_cost=calc_execution_costs(
                    total_tokens,
                    cost_per_1000_tokens
                ),
                total_tokens=total_tokens,
            )
        # Get result from response (remove all text before) and execute it in ipython
        else:
            result = response
            # Split result into separate commands by "In [${number}]: " using regex
            commands = re.split(r"In \[\d+\]: ", result)
            for command in commands:
                is_pip_install = "pip install" in command
                command = sanitize_command(command)
                print(f'IPYTHON REQUEST: {command}')
                c.sendline(command)
                execution_number += 1
                try:
                    c.expect_exact(f'In [{execution_number}]:')
                except pexpect.exceptions.TIMEOUT:
                    return Response(
                        final_prompt=prompt,
                        answer='Failed to execute command. Timeout',
                        iterations=iteration,
                        execution_total_cost=calc_execution_costs(
                            total_tokens,
                            cost_per_1000_tokens
                        ),
                        total_tokens=total_tokens,
                    )
                execute_result = sanitize_stdout(c.before)
                if is_pip_install:
                    execute_result = execute_result.replace(
                        'Note: you may need to '
                        'restart the kernel to use '
                        'updated packages.\r',
                        ''
                    )
                prompt += execute_result
                if 'Out[' not in prompt.split('\n')[-1]:
                    prompt += sanitize_stdout(c.after)
                else:
                    break
                # Break execution sequence result contains
                # "Traceback (most recent call last)"
                if 'Traceback (most recent call last)' in execute_result:
                    break
        if iteration > max_iterations:
            break
        iteration += 1
    return Response(
        final_prompt=prompt,
        answer='Can not find answer. Please, try again with another question',
        iterations=iteration,
        execution_total_cost=calc_execution_costs(
            total_tokens,
            cost_per_1000_tokens
        ),
        total_tokens=total_tokens,
    )

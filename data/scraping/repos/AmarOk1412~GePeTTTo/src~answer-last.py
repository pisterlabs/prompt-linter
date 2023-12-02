#!/usr/bin/env python3

import cmd
import json
import openai
import os
import re

from gitparser import GitParser
from termcolor import colored, cprint

class GePeTTTo(cmd.Cmd):
    prompt = colored('(GePeTTTo) ', 'yellow')
    file = None
    intro = colored(f'Type help or ? to list commands.\n', 'green')
    endpoint = os.getenv('GITLAB_ENDPOINT')
    parser = GitParser(endpoint)
    issues = []
    answeringTo = None
    answer = ''
    model = os.getenv('OPENAI_MODEL')

    def answerTo(self, issue):
        '''Answer to an issue via GPT-3 model'''
        self.answer = openai.Completion.create(
            model=self.model,
            prompt=f'Answer to this issue:\n\n{json.dumps(issue["body"])}',
            temperature=0, max_tokens=500,
            stop='<END>'
        )['choices'][0]['text']

    def do_help(self, args=''):
        print('start: Start to analyze issues')
        print('save <file>: Save generated answer to <file>')
        print('send: Send answer without saving it')
        print('send <file>: Send answer from file')
        print('add <url>: Add an issue from an url')
        print('add <file>: Add an issue from a file (no send possible)')
        print('next: Pass to the next issue')
        print('exit: Close GePeTTTo')

    def do_clear(self, arg):
        '''Clear your prompt: CLEAR'''
        os.system('clear')

    def do_add(self, arg):
        '''Start the analysis'''
        if arg.startswith('http'):
            match = re.match(f'{self.endpoint}/(\w+)/(.+)/-/issues/([0-9]+)', arg)
            if match:
                project_id = 0
                for pj in self.parser.get_projects():
                    if pj['name'] == match[2]:
                        project_id = pj['id']
                        break
                if project_id == 0:
                    print('No project found')
                    return
                self.issues.append(self.parser.get_issue(project_id, match[3]))
                print('Generating answer...')
                self.do_next('')
                return
            print('Incorrect url')
        else:
            # Try to add the issue from a file
            with open(arg, 'r') as f:
                self.issues.append({'id': 0,
                                    'project_id': 0,
                                    'author': 'NONE',
                                    'body': f.read()})
                print('Generating answer...')
                self.do_next('')


    def do_EOF(self, arg):
        return True

    def do_start(self, arg):
        '''Start the analysis'''
        print('Retrieving issues... (can be slow)')
        self.issues = self.parser.get_last_issues()
        self.do_next(arg)

    def do_next(self, arg):
        '''Show the next issue to analyze'''
        if self.answeringTo != None and self.answeringTo['id'] != 0:
            # Save last answered issue
            self.parser.set_last_issue(self.answeringTo['id'])
        if len(self.issues) == 0:
            print('No issue to analyze (you can add one or exit)')
            return
        self.answeringTo = self.issues[0]
        self.answerTo(self.answeringTo)
        self.print_issue(self.answeringTo, self.answer)
        self.issues = self.issues[1:]
        print(self.intro)

    def do_send(self, arg):
        '''Send answer to GitLab'''
        if self.answeringTo['id'] == 0:
            print('Cannot send answer to GitLab (issue from a file). You can use the "save" command')
            return
        try:
            if arg != '':
                with open(arg, 'r') as f:
                    self.answer = f.read()
            self.parser.answer(self.answeringTo['id'], self.answeringTo['project'], self.answer)
        except:
            pass
        self.do_next('')

    def do_save(self, arg):
        '''Save generated answer to a file'''
        filename = arg if arg != '' else f'{self.answeringTo["id"]}.answer'
        with open(filename, 'w+') as f:
            f.write(self.answer)

    def do_exit(self, arg):
        '''Close GePeTTTo'''
        exit(0)

    def print_issue(self, issue, answer):
        '''Display issue + generated answer'''
        self.do_clear('')
        text = colored(f'Answer to Issue #{issue["id"]} (by {issue["author"]}) - Description:\n', 'green')
        text += colored(issue["body"], 'white')
        text += colored(f'''

====================================================

Proposed answer from GePeTTTo:
''', 'green')
        text += colored(answer, 'white')
        print(text)


if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    gepettto = GePeTTTo()
    gepettto.cmdloop()
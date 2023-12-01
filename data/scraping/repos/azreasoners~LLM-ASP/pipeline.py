import os
import pickle
import time

from clingo.control import Control
from clingo.symbol import parse_term
import pandas as pd
import openai

# Enter your GPT-3 API key here
API_KEY = 'XXX'
# [optional] you may also put your ORG key below
ORG_KEY = ''

class Context:
    def gen_feature(self, x):
        ret = []
        for term in str(x.string).split(' '):
            ret.append(parse_term(term))
        return ret

class Pipeline:
    def __init__(self, args):
        self.asp_program = ''
        ###########
        # GPT-3
        ###########
        self.org_key = ORG_KEY
        self.api_key = API_KEY
        self.engine = 'text-davinci-003'
        self.temperature = 0.
        self.max_tokens = 256
        self.prompt = {} # a mapping from prompt kind (str) to the prompt (str)
        self.count = 0 # a counter for the number of GPT-3 query
        ###########
        # Cache
        ###########
        self.path_cache = {} # store the mapping from kind (str) to cache file (str)
        self.cache = {} # store the GPT3 responses for visited stories
        self.path_mistakes = 'mistakes.xlsx' # file to store the wrong pridictions
        self.mistakes = [] # store the wrong predictions

        for k,v in args.items():
            setattr(self, k, v)
        # init openai account
        if self.org_key:
            openai.organization = self.org_key
        openai.api_key = self.api_key

    def load_prompt(self, kind_to_path):
        for kind in kind_to_path:
            with open(kind_to_path[kind], 'r', encoding='utf-8') as f:
                self.prompt[kind] = f.read()

    def load_cache(self):
        for kind in self.path_cache:
            if os.path.isfile(self.path_cache[kind]):
                with open(self.path_cache[kind], 'rb') as f:
                    self.cache[kind] = pickle.load(f)
            else:
                self.cache[kind] = {}
    
    def save_cache(self):
        for kind in self.path_cache:
            with open(self.path_cache[kind], 'wb') as f:
                pickle.dump(self.cache[kind], f, protocol=pickle.HIGHEST_PROTOCOL)
            if self.count % 100 == 0:
                with open(self.path_cache[kind]+str(self.count), 'wb') as f:
                    pickle.dump(self.cache[kind], f, protocol=pickle.HIGHEST_PROTOCOL)

    # take a sentence and its kind, return the GPT3 response
    def gen_response(self, sentence, kind):
        # generate and cache the response in cache if it's not cached before
        if sentence not in self.cache[kind]:
            # print('==== Call GPT-3 ====')
            self.count += 1
            # print out the counting for every 100 queries
            if self.count % 100 == 0:
                print(self.count)
            time.sleep(2)
            try:
                self.cache[kind][sentence] = openai.Completion.create(
                    prompt=self.prompt[kind] + sentence.strip() + '\nSemantic Parse:',
                    engine=self.engine,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens)
                self.save_cache()
            except Exception as e:
                print(e)
                self.cache[kind][sentence] = None

        # obtain the response from cache
        response = ''
        if self.cache[kind][sentence] is not None:
            response = self.cache[kind][sentence]['choices'][0]['text'].strip()
        # stop if response is empty due to content filtering or other issue
        assert response != '', 'Error: GPT-3 response is empty'
        return response
    
    # take a list of GPT3 responses and return the answer set
    def gen_answer_set(self, responses, opt=False):
        """
        Args:
            responses (str): a string of ASP facts
            opt (bool): if true, only optimal answer sets are returned
                        leave it to False when there is no weak constraint
        """
        program = self.asp_program + responses
        clingo_control = Control(['0', '--warn=none', '--opt-mode=optN', '-t', '4'])
        models = []
        try:
            clingo_control.add('base', [], program)
            clingo_control.ground([('base', [])], context=Context())
        except:
            if self.debug:
                print(responses)
                breakpoint()
            return []
        if opt:
            clingo_control.solve(on_model = lambda model: models.append(model.symbols(atoms=True)) if model.optimality_proven else None)
        else:
            clingo_control.solve(on_model = lambda model: models.append(model.symbols(atoms=True)))
        models = [[str(atom) for atom in model] for model in models]
        return models

    # return the answer sets for an example data instance
    def eval_single_example(self, example, facts='', clean=False, opt=False):
        """
        Args:
            example (dict): a mapping from kind (str) to a list of sentences
            facts (str): additionally facts used during inference
            clean (bool): if true, remove all quotes turn all letters into lower case
        """
        response = ''
        for kind in example:
            for sentence in example[kind]:
                response += self.gen_response(sentence, kind)
        if clean:
            response = response.replace('\"', '').lower()
        # any dash between words should be replaced by underscore in clingo
        response = response.replace('-', '_')
        answer_sets = self.gen_answer_set(response + '\n\n' + facts, opt=opt)
        return answer_sets, response

    def save_mistakes(self, mistake_cols):
        df = pd.DataFrame(self.mistakes, columns=mistake_cols)
        df.to_excel(self.path_mistakes)


class Baseline(Pipeline):
    # take the name of a prompt context and an [INPUT], return the GPT3 response
    def gen_response(self, name, inp):
        # generate and cache the response if it's not cached before
        if inp not in self.cache[name]:
            # print('==== Call GPT-3 ====')
            self.count += 1
            # print out the counting for every 100 queries
            if self.count % 100 == 0:
                print(self.count)
            time.sleep(2)
            try:
                self.cache[name][inp] = openai.Completion.create(
                    prompt=self.prompt[name].replace('[INPUT]', inp),
                    engine=self.engine,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens)
                self.save_cache()
            except Exception as e:
                print(e)
                self.cache[name][inp] = None

        # obtain the response from cache
        response = ''
        if self.cache[name][inp] is not None:
            response = self.cache[name][inp]['choices'][0]['text'].strip()
        # stop if response is empty due to content filtering or other issue
        assert response != '', 'Error: GPT-3 response is empty'
        return response

    # return the answer sets for an example data instance
    def eval_single_example(self, example):
        """
        Args:
            example (dict): a mapping from prompt name (str) to an input example (str)
        """
        response = {}
        for name in example:
            if name in self.prompt:
                response[name] = self.gen_response(name, example[name])
        return response

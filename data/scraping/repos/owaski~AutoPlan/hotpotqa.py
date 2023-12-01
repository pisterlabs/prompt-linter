import os
import time
import json 
import regex
import random
import pickle 
import re

import numpy as np
import openai
from func_timeout import func_timeout, FunctionTimedOut
from revChatGPT.V1 import Chatbot
import wandb
from tqdm import tqdm

from envs.wikienv import WikiEnv
from scorers import hotpotqa_eval
from tasks.utils import PRICE, pretty_print

class HotpotQA:
    def __init__(self, lm, method, backend_args, quota_args, **kwargs):
        self.lm = lm
        self.method = method
        self.backend = backend_args['name']
        
        if self.backend == 'openai':
            with open(backend_args['api_key'], 'r') as r:
                openai.api_key = r.read()
            if backend_args['org_id'] is not None:
                with open(backend_args['org_id'], 'r') as r:
                    openai.organization = r.read()
            self.top_p = backend_args['top_p']
            self.temp = backend_args['temp']
            self.max_token = backend_args['max_token']
            self.presence_penalty = backend_args['presence_penalty']
        elif self.backend == 'revChatGPT':
            with open(backend_args['access_token'], 'r') as r:
                self.access_token = r.read()
        
        self.max_budget = quota_args['max_budget']
        self.max_iter_per_instance = quota_args['max_iter_per_instance']

        # self.message_counter = 0

        self.wiki = WikiEnv()
        self.load_data()

        self.history = []
        self.strategy = None

        # openai api
        self.n_prompt_token = 0
        self.n_sample_token = 0
        self.messages = []

    
    def task_desription(self):
        return '''Solve a question answering task through actions which can be of three types: 
(1) search[entity]: search the entity on Wikipedia. If there is an exact match of the entity, open its Wikipedia page and return the first paragraph. If not, it will return some related entities.
(2) lookup[keyword]: return the next sentence containing the keyword in the current Wikipedia page opened by search[].
(3) finish[answer]: return the answer and finish the task.'''

    def load_data(self):
        # hard coded for now
        data_root = 'path_to_hotpotqa'
        self.data = {}
        with open(os.path.join(data_root, 'hotpot_train_v1.1_filtered.json'), 'r') as r:
            self.data['train'] = json.load(r)
        with open(os.path.join(data_root, 'hotpot_dev_distractor_v1_filtered.json'), 'r') as r:
            self.data['dev'] = json.load(r)

        
    def collate_fn(self, data):
        return data

    
    def score(self, instance, prediction):
        question, answer = instance['question'], instance['answer']
        query = """Question: "{}"\n
Gold Answer: "{}"\n
My Prediction: "{}"\n
Verify whether my prediction to the question is equivalent to gold answer. Respond with yes/no.""".format(question, answer, prediction)

        message = {
            'role': 'user',
            'content': query
        }
        
        while True:
            try:
                response = func_timeout(10, 
                    openai.ChatCompletion.create,
                    kwargs={
                        "model": "gpt-4-0314",
                        "messages": [message],
                        "temperature": 0.
                    }                
                )['choices'][0]['message']['content'].lower()
                break
            except FunctionTimedOut:
                print('OpenAI API call timeout')
                continue
        
        assert 'yes' in response or 'no' in response, (question, answer, prediction, response)
        return 'yes' in response
        

    def eval(self, instances, predictions):
        n_correct = 0
        correct_mask = []
        for idx in tqdm(range(len(instances)), desc='Evaluating'):
            result = self.score(instances[idx], predictions[idx])
            n_correct += int(result)
            correct_mask.append(result)
        
        return {'n_correct': n_correct, 'correct_mask': correct_mask}


    def formalize(self, action):
        message = """Valid action formats are as follows:
(1) search[entity]
(2) lookup[keyword]
(3) finish[answer]

Formalize the following action strictly into the above valid action formats. If there are multiple actions, formalize the first one.

Action: I want to search the entity "Laptop".
Formalized First Action: search[Laptop]

Action: look up the word "desk".
Formalized First Action: lookup[desk]

Action: {}
Formalized First Action:
""".format(action)
        message = {'role': 'user', 'content': message}
        action = self.call_openai_api([message], stop='\n')['choices'][0]['message']['content']

        return action


    def recognize(self, response):
        # recognize the first occurence of string "search(entity)" and "lookup(string)" 
        # and "Finalize[answer]" in response by regex and return the string

        # response = response.replace('search[entity]', ' ')
        # response = response.replace('lookup[keyword]', ' ')
        # response = response.replace('finish[answer]', ' ')

        search = regex.search(r'search\[(.*)\]', response)
        lookup = regex.search(r'lookup\[(.*)\]', response)
        finish = regex.search(r'finish\[(.*)\]', response)

        max_start = max([x.start() for x in [search, lookup, finish] if x is not None], default=-1)
        if max_start == -1:
            match = None
        elif search and max_start == search.start():
            match = search
        elif lookup and max_start == lookup.start():
            match = lookup
        elif finish and max_start == finish.start():
            match = finish
        else:
            raise NotImplementedError
        return match, search, lookup, finish

    def price(self):
        price = PRICE[self.lm]
        return (self.n_prompt_token * price['prompt'] + self.n_sample_token * price['sample']) / 1000
    
    def call_openai_api(self, messages, stop, lm=None, top_p=None):
        n_try = 10
        while n_try > 0:
            try:
                time.sleep(1)
                response = func_timeout(90, 
                    openai.ChatCompletion.create,
                    kwargs={
                        "model": self.lm if lm is None else lm,
                        "messages": messages,
                        "top_p": self.top_p if top_p is None else top_p,
                        "temperature": self.temp,
                        "max_tokens": self.max_token,
                        "presence_penalty": self.presence_penalty,
                        "stop": stop,
                    }
                )
                break
            except FunctionTimedOut:
                print('[LOG] OpenAI API call timeout')
                n_try -= 1
                if n_try == 0:
                    raise Exception('Failed 10 retries.')
                continue
            except Exception as e:
                print('[LOG]', e)
                time.sleep(10)
                n_try -= 1
                if n_try == 0:
                    raise Exception('Failed 10 retries.')
                continue
        return response
        
    def call_lm(self, prompt, add_response=True, stop=None, lm=None, top_p=None):
        if self.backend == 'openai':
            self.messages.append({'role': 'user', 'content': prompt})
            response = self.call_openai_api(self.messages, stop, lm=lm, top_p=top_p)
            if add_response: self.messages.append(response['choices'][0]['message'])
            self.n_prompt_token += response['usage']['prompt_tokens']
            self.n_sample_token += response['usage']['completion_tokens']
            return response['choices'][0]['message']['content']
        elif self.backend == 'revChatGPT':
            try:
                prev_text = ""
                for data in self.chatbot.ask(prompt):
                    message = data["message"][len(prev_text) :]
                    prev_text = data["message"]
                return data['message']
            except Exception as e:
                if hasattr(e, 'code') and e.code == 429:
                    print(e)
                    print('Sleeping for {} minutes...'.format(self.sleep_minutes))
                    time.sleep(self.sleep_minutes * 60)
                    return self.call_lm(prompt)
                else:
                    raise e
                
    
    def init(self):
        prompt = "{}\n\nTask Plan:".format(self.task_desription())
        strategy = self.call_lm(prompt)
        return strategy
    

    def run(self, data, strategy=None, is_test=False, verbose=False, react=False, retrieval=False, return_history=False):
        questions = []
        answers = []
        predictions = []
        summaries = []
        flawed_actions = []
        flawed_plans = []
        history = ''

        if react:
            if retrieval:
                assert len(data) == 1
                question = data[0]['question']
                with open('path_to_hotpotqa/train_v1.1_filtered_retrieval.pkl', 'rb') as r:
                    succ_embeddings, historys = pickle.load(r)
                succ_embeddings = np.array(succ_embeddings)            
                embedding = openai.Embedding.create(
                    input=question,
                    model='text-embedding-ada-002'
                )['data'][0]['embedding']
                scores = np.dot(succ_embeddings, embedding)
                top6_indices = np.argsort(scores)[::-1][:6]
                icl_examples = []
                for idx in top6_indices:
                    icl_examples.append(historys[idx])
                icl_examples = '\n'.join(icl_examples)
            else:
                with open('path_to_ReAct/prompts/prompts_naive.json', 'r') as f:
                    prompt_dict = json.load(f)
                icl_examples = prompt_dict['webthink_simple6']

        for q_idx in range(len(data)):
            self.messages = []
            init_msg = self.task_desription()
            if react:
                init_msg += '\n\nHere are some examples:\n' + icl_examples
            elif 'direct' not in self.method and strategy is not None:
                init_msg += "\n\nTask Plan:\n{}\n".format(strategy)
            init_msg += '\n'

            question = data[q_idx]['question']
            answer = data[q_idx]['answer']
            questions.append(question)
            answers.append(answer)

            # print(question, answer, sep='\n')

            if 'step' in self.method and strategy is not None:
                thought_prompt = "Identify which step of plan you are at. Show your thought about the one next action. Your thought should be faithful to the plan step."
            else:
                thought_prompt = "Show your thought about the next action."

            input_msg = init_msg + '\nQuestion: ' + question + '\n' + thought_prompt
            history += pretty_print('Human', input_msg, verbose)

            cur_itr = 0
            has_answer = False
            while cur_itr < self.max_iter_per_instance:
                
                # Thought
                thought = self.call_lm(input_msg, stop='\n')
                history += pretty_print('Machine', thought, verbose)

                # Action
                history += pretty_print('Human', 'Action:', verbose)
                action = self.call_lm('Action:', add_response=False, stop='\n')
                formalized_action = self.formalize(action)
                self.messages.append({'role': 'assistant', 'content': formalized_action})
                history += pretty_print('Machine', '{} (orig {})'.format(formalized_action, action), verbose)

                rec_res, m_search, m_em, m_final = self.recognize(formalized_action)

                cmd, object = None, None
                if rec_res:
                    cmd = rec_res.group(0)
                    object = rec_res.group(1).strip('"\'')
                
                if rec_res:
                    input_msg = 'Observation:\n'
                    if rec_res == m_search:
                        self.wiki.search_step(object)
                        input_msg += self.wiki.obs
                    elif rec_res == m_em:
                        lookup_return = self.wiki.step('lookup[{}]'.format(object))[0]
                        input_msg += lookup_return
                    else:
                        predictions.append(object)
                        has_answer = True
                        break
                else:
                    input_msg = random.choice([
                        "No action in the format of 'search[...]', 'lookup[...]', or 'finish[...]'.",
                    ])

                if cur_itr < self.max_iter_per_instance - 1:
                    input_msg += '\n' + thought_prompt
                history += pretty_print('Human', input_msg, verbose)
                
                cur_itr += 1

            if not has_answer:
                predictions.append('')
            
            if not is_test:
                if not has_answer:
                    summary_msg = 'Max number of iteration reached. No answer is found.'
                else:
                    summary_msg = 'Task finished.'

                supporting_entities = set()
                for f in data[q_idx]['supporting_facts']:
                    supporting_entities.add('"{}"'.format(f[0]))
                supporting_entities = list(supporting_entities)
                
                supporting_string = None
                if len(supporting_entities) == 1:
                    supporting_string = supporting_entities[0]
                else:
                    supporting_string = ', '.join(supporting_entities[:-1]) + ' and ' + supporting_entities[-1]                

                summary_msg += ' The ground truth answer is "{}" and the correct entities to search are {}. Summarize the interaction history concisely.'.format(answer, supporting_string)

                history += pretty_print('Human', summary_msg, verbose)
                summary = self.call_lm(summary_msg, top_p=0.)
                history += pretty_print('Machine', summary, verbose)
                summaries.append(summary)
                
                if strategy is not None:

                    failed_action_msg = 'Identify all flawed parts of the plan (not flawed action). Only the flawed part.'
                    history += pretty_print('Human', failed_action_msg, verbose)
                    failed_action = self.call_lm(failed_action_msg, top_p=0.)
                    history += pretty_print('Machine', failed_action, verbose)
                    flawed_actions.append(failed_action)

                    suggest_rev_msg = 'Suggest revision to the current flawed part of the plan. Only the flawed part.'
                    history += pretty_print('Human', suggest_rev_msg, verbose)
                    suggest_rev = self.call_lm(suggest_rev_msg, stop=None, top_p=0.0)
                    history += pretty_print('Machine', suggest_rev, verbose)
                    flawed_plans.append(suggest_rev)

        to_return = None

        if is_test:
            to_return = predictions
        else:
            self.messages = []
            final_msg = 'Task Description:\n' + self.task_desription() + '\n\n'
            final_msg += 'Current Task Plan:\n{}\n\n'.format(strategy)
            final_msg += '=' * 10 + 'Task Experiences Begin' + '=' * 10 + '\n\n'

            
            for q_idx in range(len(data)):
                question = data[q_idx]['question']
                final_msg += 'Job {}: Answering the following question. {}\nSummary of Job {}:\n{}\n'.format(q_idx, question, q_idx, summaries[q_idx])
                if strategy is not None:
                    final_msg += 'Flaws of Plan in Job {}:\n{}\n'.format(q_idx, flawed_actions[q_idx])
                    final_msg += 'Suggested Revision of Plan from Job {}:\n{}\n'.format(q_idx, flawed_plans[q_idx])

            final_msg += '=' * 10 + 'Task Experiences End' + '=' * 10 + '\n\n'

            final_msg += 'Based on the above {} experiences of the task, rewrite the current task plan. The plan should not be specific to one job but generalizable to all jobs. \nNew Task Plan:'.format(len(data))

            history += pretty_print('Human', final_msg, verbose)
            new_strategy = self.call_lm(final_msg, top_p=0.)
            history += pretty_print('Machine', new_strategy, verbose)

            to_return = new_strategy

        self.history.append(history)

        if return_history:
            to_return = to_return, history

        if not is_test:
            n_correct = self.eval(data, predictions)['n_correct']
            wandb.log({"train_acc": n_correct / len(questions)})

        return to_return
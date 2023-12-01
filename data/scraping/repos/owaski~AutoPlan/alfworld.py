import os
import time
import openai
import json
import random
import wandb
import re
import difflib
from string import punctuation
from func_timeout import func_timeout, FunctionTimedOut

os.environ['ALFWORLD_DATA'] = 'path_to_alfworld'

import torch
import numpy as np
import editdistance
from sentence_transformers import SentenceTransformer, util

import yaml
import alfworld
import alfworld.agents.environment

from tqdm import tqdm

import textworld
import textworld.agents
import textworld.gym
import gym
from alfworld.agents.utils.misc import Demangler

from tasks.utils import PRICE, pretty_print

TASKS = [
    'pick_and_place',
    'look_at_obj',
    'pick_clean_then_place',
    'pick_heat_then_place',
    'pick_cool_then_place',
    'pick_two_obj'
]

SEEDS = list(range(1000))

REACT_PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

GOAL_DESCS = {
    'pick_and_place': 'Put some object in/on some receptacle.',
    'look_at_obj': 'Examine some object with the lamp.',
    'pick_clean_then_place': 'Clean some object and put it in/on some receptacle.',
    'pick_heat_then_place': 'Heat some object and put it in/on some receptacle.',
    'pick_cool_then_place': 'Cool some object and put it in/on some receptacle.',
    'pick_two_obj': 'Find two objects and put them in/on some receptacles.'
}

class AlfredDemangler(textworld.core.Wrapper):

    def __init__(self, shuffle=False):
        super().__init__()
        self.shuffle = shuffle

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)

        demangler = Demangler(game_infos=self._game.infos, shuffle=self.shuffle)
        for info in self._game.infos.values():
            info.name = demangler.demangle_alfred_name(info.id)


class AlfredInfos(textworld.core.Wrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gamefile = None

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)
        self._gamefile = args[0]

    def reset(self, *args, **kwargs):
        state = super().reset(*args, **kwargs)
        state["extra.gamefile"] = self._gamefile
        return state
    

def get_env_with_idx(env, idx):
    batch_size = 1
    training_method = env.config["general"]["training_method"]
    expert_type = env.config["env"]["expert_type"]
    if training_method == "dqn":
        infos = textworld.EnvInfos(won=True, admissible_commands=True, expert_type=expert_type, expert_plan=False, extras=["gamefile"])
        max_nb_steps_per_episode = env.config["rl"]["training"]["max_nb_steps_per_episode"]
    elif training_method == "dagger":
        expert_plan = True if env.train_eval == "train" else False
        infos = textworld.EnvInfos(won=True, admissible_commands=True, expert_type=expert_type, expert_plan=expert_plan, extras=["gamefile"])
        max_nb_steps_per_episode = env.config["dagger"]["training"]["max_nb_steps_per_episode"]
    else:
        raise NotImplementedError

    domain_randomization = env.config["env"]["domain_randomization"]
    if env.train_eval != "train":
        domain_randomization = False
    alfred_demangler = AlfredDemangler(shuffle=domain_randomization)
    env_id = textworld.gym.register_games([env.game_files[idx]], infos,
                                            batch_size=batch_size,
                                            asynchronous=True,
                                            max_episode_steps=max_nb_steps_per_episode,
                                            wrappers=[alfred_demangler, AlfredInfos])
    # launch Gym environment.
    new_env = gym.make(env_id)
    return new_env


class ALFWorld:
    def __init__(self, lm, method, backend_args, quota_args, subtask=None, train_bsz=1, random_explore=0.0, **kwargs):
        self.subtask = subtask
        self.lm = lm
        self.method = method
        self.backend = backend_args['name']

        assert self.backend == 'openai'
        with open(backend_args['api_key'], 'r') as r:
            openai.api_key = r.read()
        if backend_args['org_id'] is not None:
            with open(backend_args['org_id'], 'r') as r:
                openai.organization = r.read()
        self.top_p = backend_args['top_p']
        self.temp = backend_args['temp']
        self.max_token = backend_args['max_token']
        self.presence_penalty = backend_args['presence_penalty']

        self.max_budget = quota_args['max_budget']
        self.max_iter_per_instance = quota_args['max_iter_per_instance']

        with open('base_config.yaml') as reader:
            self.config = yaml.safe_load(reader)
        self.config['env']['task_types'] = [TASKS.index(self.subtask) + 1]
        self.load_data(train_bsz)

        self.history = []
        self.strategy = None

        # openai api
        self._price = 0
        self.n_prompt_token = 0
        self.n_sample_token = 0
        self.messages = []

        # sentence embedding
        self.random_explore = random_explore

        # state
        self.visited_locations = set()
        
        
    def load_data(self, bsz):
        self.data = {}

        self.data['train'] = []
        train_env = getattr(alfworld.agents.environment, self.config["env"]["type"])(self.config, train_eval='train')
        for i in range(len(train_env.game_files)):
            self.data['train'].append(get_env_with_idx(train_env, i))


        self.data['dev'] = []
        dev_env = getattr(alfworld.agents.environment, self.config["env"]["type"])(self.config, train_eval='eval_out_of_distribution')
        for i in range(len(dev_env.game_files)):
            self.data['dev'].append(get_env_with_idx(dev_env, i))

    
    def get_data(self, split, idx):
        assert split == 'dev'
        return self.data[split].reset(idx)

    
    def score(self, instance, prediction):
        return prediction



# (1) go_to(recep, j): go to recep j
# (2) take(object, i, recep, j): take object i from recep j and carry it with you
# (3) put(object, i, recep, j): put the object i you are carrying in/on recep j
# (4) open(recep, j): open some recep j
# (5) close(recep, j): close recep j
# (6) use(recep, j): use recep j
# (7) clean(object, i, recep, j): clean object i with recep j
# (8) heat(object, i, recep, j): heat object i with recep j
# (9) cool(object, i, recep, j): cool object i with recep j

    def task_description(self):
        return r'''You need to interact with a simulated household to solve a job. The simulated house has many objects and receptacles. Valid Actions on the objects and receptacles are as follows:
(1) go to recep j
(2) take object i from recep j
(3) put object i in/on recep j
(4) open recep j
(5) close recep j
(6) use recep j
(7) clean object i with recep j
(8) heat object i with recep j
(9) cool object i with recep j
You job is to {} 
'''.format(GOAL_DESCS[self.subtask])


    def collate_fn(self, data):
        return data
    

    def update_price(self, usage, lm):
        self._price += usage['prompt_tokens'] * PRICE[lm]['prompt'] / 1000
        self._price += usage['completion_tokens'] * PRICE[lm]['sample'] / 1000


    def price(self):
        return self._price

    def call_openai_api(self, messages, stop, lm=None, top_p=None, temp=None, max_token=None):
        n_try = 10
        while n_try > 0:
            try:
                time.sleep(1)
                if not 'text' in self.lm:
                    response = func_timeout(75, 
                        openai.ChatCompletion.create,
                        kwargs={
                            "model": self.lm if lm is None else lm,
                            "messages": messages,
                            "top_p": self.top_p if top_p is None else top_p,
                            "temperature": self.temp if temp is None else temp,
                            "max_tokens": self.max_token if max_token is None else max_token,
                            "presence_penalty": self.presence_penalty,
                            "stop": stop,
                        }
                    )
                    self.update_price(response['usage'], self.lm if lm is None else lm)
                    response = response['choices'][0]['message']['content']
                else:
                    prompt = '\n'.join([m['content'] for m in messages])
                    response = func_timeout(45, 
                        openai.Completion.create,
                        kwargs={
                            "model": self.lm,
                            "prompt": prompt,
                            "top_p": self.top_p,
                            "temperature": self.temp,
                            "max_tokens": self.max_token,
                            "presence_penalty": self.presence_penalty,
                            "stop": stop,
                        }
                    )
                    self.n_prompt_token += response['usage']['prompt_tokens']
                    response = response['choices'][0]['text']
                break
            except FunctionTimedOut:
                print('[LOG] OpenAI API call timeout')
                time.sleep(10)
                n_try -= 1
                if n_try == 0:
                    raise Exception('Failed 10 retries.')
                continue
            except Exception as e:
                if 'This model\'s maximum context length is' in e._message:
                    raise e
                print('[LOG]', e)
                time.sleep(10)
                n_try -= 1
                if n_try == 0:
                    raise Exception('Failed 10 retries.')
                continue
        return response
        
    
    def call_lm(self, prompt, stop=None, lm=None, top_p=None, temp=None, max_token=None):
        self.messages.append({'role': 'user', 'content': prompt})
        # total_message = '\n'.join([m['content'] for m in self.messages])
        response = self.call_openai_api(self.messages, stop, lm=lm, top_p=top_p, temp=temp, max_token=max_token)
        self.messages.append({'role': 'assistant', 'content': response})
        return response

    
    def error_msg(self, action):
        re_obj_idx = '([^\d]+ \d+)'
        re_obj = '([^\d]+( \d+|))'
        err_msg = ""

        def obj_fmt_err(obj):
            obj = obj.strip()
            obj_idx = obj.split(' ')
            if len(obj_idx) == 1:
                return 'The index of {} is missing. For example, {} 0.'.format(obj, obj)
            if len(obj_idx) > 2:
                return 'The format of {} is wrong. '.format(obj)
            return ''
        
        def at_recep_err(recep):
            recep = recep.strip()
            if self._state['location'] != recep:
                return 'You are not at {}. '.format(recep)
            return ''

        def exist_recep_err(recep):
            recep = recep.strip()
            if recep not in self._state['all locations']:
                return '{} is not a valid receptacle in this household. '.format(recep)
            return ''

        if 'go to ' in action:
            match = re.fullmatch('go to {}'.format(re_obj), action)
            if match is None:
                return 'System Message: The action should be in the form of "go to receptacle i". '

            recep = match.group(1)
            err_msg += obj_fmt_err(recep)

            if err_msg == '':
                err_msg += exist_recep_err(recep)
                if err_msg == '':
                    err_msg += 'You are already at {}. '.format(recep)
            
        elif 'take' in action:
            match = re.fullmatch('take {} from {}'.format(re_obj, re_obj), action)
            if match is None:
                return 'System Message: The action should be in the form of "take object i from receptacle j". '
            obj, recep = match.group(1), match.group(3)

            err_msg += obj_fmt_err(obj)
            err_msg += obj_fmt_err(recep)
            
            if err_msg == "":
                
                if exist_recep_err(recep) != "":
                    err_msg += 'You cannot take {} from {}. '.format(obj, recep)
                elif at_recep_err(recep) != "" and exist_recep_err(recep) == "":
                    err_msg += 'You are not at {}. '.format(recep)
                else:
                    if self._state['status'][recep] == 'closed':
                        err_msg += '{} is closed. '.format(recep)
                    elif obj not in self._state['contain'][recep]:
                        err_msg += '{} is not in {}. '.format(obj, recep)
                
                if len(self._state['inventory']) >= 1:
                    err_msg += 'You cannot hold more than one object. '

            if err_msg == "":
                err_msg += 'You cannot take {} from {}. '.format(obj, recep)
        
        elif 'put' in action:
            match = re.fullmatch('put {} in/on {}'.format(re_obj, re_obj), action)
            if match is None:
                return 'System Message: The action should be in the form of "put object i in/on receptacle j" .'
            obj, recep = match.group(1), match.group(3)
            
            err_msg += obj_fmt_err(obj)
            err_msg += obj_fmt_err(recep)

            if err_msg == "":
                if exist_recep_err(recep) != "":
                    err_msg += 'You cannot put {} in/on {}. '.format(obj, recep)
                elif at_recep_err(recep) != "" and exist_recep_err(recep) == "":
                    err_msg += 'You are not at {}. '.format(recep)
                else:
                    if self._state['status'][recep] == 'closed':
                        err_msg += '{} is closed. '.format(recep)
                
                if obj not in self._state['inventory']:
                    err_msg += 'You are not carrying {}. '.format(obj)
        
        elif 'open' in action:
            match = re.fullmatch('open {}'.format(re_obj), action)
            if match is None:
                return 'System Message: The action should be in the form of "open receptacle i". '
            recep = match.group(1)

            err_msg += obj_fmt_err(recep)

            if err_msg == "":
                err_msg += at_recep_err(recep)
                err_msg += exist_recep_err(recep)

                if err_msg == '':
                    if self._state['status'][recep] == 'open':
                        err_msg += '{} is already open. '.format(recep)
                    else:
                        err_msg += '{} is not openable. '.format(recep)
            
        elif 'close' in action:
            match = re.fullmatch('close {}'.format(re_obj), action)
            if match is None:
                return 'System Message: The action should be in the form of "close receptacle i". '
            recep = match.group(1)

            err_msg += obj_fmt_err(recep)

            if err_msg == "":
                err_msg += at_recep_err(recep)
                err_msg += exist_recep_err(recep)

                if err_msg == '':
                    if self._state['status'][recep] == 'closed':
                        err_msg += '{} is already closed. '.format(recep)
                    else:
                        err_msg += '{} is not closable. '.format(recep)
            
        elif 'use' in action:
            match = re.fullmatch('use {}'.format(re_obj), action)
            if match is None:
                return 'System Message: The action should be in the form of "use receptacle i". '
            recep = match.group(1)

            err_msg += obj_fmt_err(recep)

            if err_msg == "":
                if 'middle' in self._state['location'] or (self._state['status'][self._state['location']] == 'open' and recep not in self._state['contain'][self._state['location']] and recep != self._state['location']):
                    err_msg += '{} is not in/on your current location {}. '.format(recep, self._state['location'])
                else:
                    err_msg += ' In this game you cannot apply "use" action to {}. '.format(recep)
                    if 'microwave' in recep:
                        err_msg += 'You can use "heat" action to heat object with microwave. '
                    elif 'sinkbasin' in recep:
                        err_msg += 'You can use "clean" action to clean object with sinkbasin. '
                    elif 'fridge' in recep:
                        err_msg += 'You can use "cool" action to cool object with fridge. '
        
        elif 'clean' in action:
            match = re.fullmatch('clean {} with {}'.format(re_obj, re_obj), action)
            if match is None:
                return 'System Message: The action should be in the form of "clean object i with object j". '
            obj, recep = match.group(1), match.group(3)

            err_msg += obj_fmt_err(obj)
            err_msg += obj_fmt_err(recep)

            if err_msg == "":

                if exist_recep_err(recep) != "":
                    err_msg += 'You cannot clean {} with {}. '.format(obj, recep)
                elif at_recep_err(recep) != "" and exist_recep_err(recep) == "":
                    err_msg += 'You are not at {}. '.format(recep)
                else:
                    # if self._state['status'][recep] == 'closed':
                    #     err_msg += '{} is closed. '.format(recep)
                    if obj not in self._state['inventory']:
                        err_msg += 'You need to carry {} to clean it. '.format(obj)
                    if 'sinkbasin' not in recep:
                        err_msg += '{} cannot be used for cleaning. '.format(recep.split(' ')[0])
                    if err_msg == "":
                        err_msg += '{} cannot be cleaned. '.format(obj)
        
        elif 'heat' in action:
            match = re.fullmatch('heat {} with {}'.format(re_obj, re_obj), action)
            if match is None:
                return 'System Message: The action should be in the form of "heat object i with object j". '
            obj, recep = match.group(1), match.group(3)

            err_msg += obj_fmt_err(obj)
            err_msg += obj_fmt_err(recep)

            if err_msg == "":
                if exist_recep_err(recep) != "":
                    err_msg += 'You cannot heat {} with {}. '.format(obj, recep)
                elif at_recep_err(recep) != "" and exist_recep_err(recep) == "":
                    err_msg += 'You are not at {}. '.format(recep)
                else:
                    # if self._state['status'][recep] == 'closed':
                    #     err_msg += '{} is closed. '.format(recep)
                    if obj not in self._state['inventory']:
                        err_msg += 'You need to carry {} to heat it. '.format(obj)
                    if 'microwave' not in recep:
                        err_msg += '{} cannot be used for heating. '.format(recep.split(' ')[0])
                    if err_msg == "":
                        err_msg += '{} cannot be heated. '.format(obj)
        
        elif 'cool' in action:
            match = re.fullmatch('cool {} with {}'.format(re_obj, re_obj), action)
            if match is None:
                return 'System Message: The action should be in the form of "cool object i with object j". '
            obj, recep = match.group(1), match.group(3)

            err_msg += obj_fmt_err(obj)
            err_msg += obj_fmt_err(recep)

            if err_msg == "":
                if exist_recep_err(recep) != "":
                    err_msg += 'You cannot cool {} with {}. '.format(obj, recep)
                elif at_recep_err(recep) != "" and exist_recep_err(recep) == "":
                    err_msg += 'You are not at {}. '.format(recep)
                else:
                    # if self._state['status'][recep] == 'closed':
                    #     err_msg += '{} is closed. '.format(recep)
                    if obj not in self._state['inventory']:
                        err_msg += 'You need to carry {} to cool it. '.format(obj)
                    if 'fridge' not in recep:
                        err_msg += '{} cannot be used for cooling. '.format(recep.split(' ')[0])
                    if err_msg == "":
                        err_msg += '{} cannot be cooled. '.format(obj)
        else:
            err_msg += 'Invalid action format. Check the game description for the right formats.'

        if err_msg != "": err_msg = "System Message: " + err_msg
        return err_msg
    

    def filter_actions(self, actions):
        return [a for a in actions if 'examine' not in a and 'look' not in a and 'inventory' not in a]


    def init_state(self, ob, avail_actions):
        match = re.match('You are in (.+)\. Looking quickly around you, you see (.+)\.\nYour job is to: (.+)\.', ob)
        location = match.group(1).strip()
        items = match.group(2).strip()
        self.objective = match.group(3).strip()
        locations = [a[6:] for a in avail_actions if a.startswith('go to')]
        self.state = """You arrive at {}.
You have not been to {}.
You are taking nothing with you.""".format(location, ', '.join(locations))
        self._state = {'location': 'middle of a room', 'inventory': [], 'status': {}, 'contain': {}, 'all locations': locations}


    def update_state(self, action, observation, avail_actions):

#         prompt = """Act as a state updater for a game environment.
# [Current State]
# {}
# [State Update Request]
# new action:{}
# new observation:{}
# [Update Rule]
# 1. keep the format the same.
# 2. add/remove object to what you are carrying if the action is "take ... from ..." or "put ... in/on ...".
# 3. status of visited receptacles: When open/close a receptacle, update in the form "you see ... is open/closed". When clean/heat/cool/use an object, update in the form "... is clean/heated/cool/turned on". If in the observation you see what a receptacle contains, add such information in the format of "in ..., you see ...". Multiple information should be separated by newlines.
# [New State]""".format(self.state, action, observation, ', '.join(locations))
#         message = {'role': 'user', 'content': prompt}
#         _state = self.call_openai_api([message], lm='gpt-3.5-turbo', stop='[', top_p=0.0)
        
#         prompt += '\n' + _state + '\n\nTake a careful look at the new state, rewrite the corrected new state below that has no hallucination about unknown information.\n[New State]'
#         message = {'role': 'user', 'content': prompt}
#         _state_2 = self.call_openai_api([message], lm='gpt-3.5-turbo', stop=None, top_p=0.0)
#         self.state = _state_2.lower()



        if 'go to' in action:
            loc = re.findall('go to (.+)', action)[0]
            self._state['location'] = loc
            self.visited_locations.add(loc)
            if 'is open' in observation:
                recep, objects = re.findall('The (.+) is open\. In it, you see (.+)\.', observation)[0]
                self._state['status'][recep] = 'open'
                objects = objects.replace('and ', '')
                self._state['contain'][recep] = [] if 'nothing' in objects else [o[2:] for o in objects.split(', ')]
            elif 'is closed' in observation:
                recep = re.findall('The (.+) is closed\.', observation)[0]
                self._state['status'][recep] = 'closed'
            else:
                assert 'you see' in observation
                recep, objects = re.findall('.+ the (.+), you see (.+)\.', observation)[0]
                objects = objects.replace('and ', '')
                self._state['status'][recep] = 'open'
                self._state['contain'][recep] = [] if 'nothing' in objects else [o[2:] for o in objects.split(', ')]
        elif 'take' in action:
            object, recep = re.findall('take (.+) from (.+)', action)[0]
            self._state['inventory'].append(object)
            assert len(self._state['inventory']) <= 1
            if recep in self._state['contain'] and object in self._state['contain'][recep]:
                self._state['contain'][recep].remove(object)
        elif 'put' in action:
            object, recep = re.findall('put (.+) in/on (.+)', action)[0]
            assert object in self._state['inventory']
            self._state['inventory'].remove(object)
            if recep not in self._state['contain']:
                self._state['status'][recep] = 'open'
                self._state['contain'][recep] = []
            self._state['contain'][recep].append(object)
        elif 'open' in action:
            recep = re.findall('open (.+)', action)[0]
            self._state['status'][recep] = 'open'
            objects = re.findall('In it, you see (.+)\.', observation)[0]
            objects = objects.replace('and ', '')
            self._state['contain'][recep] = [] if 'nothing' in objects else [o[2:] for o in objects.split(', ')]
        elif 'close' in action:
            recep = re.findall('close (.+)', action)[0]
            self._state['status'][recep] = 'closed'
        elif 'use' in action:
            recep = re.findall('use (.+)', action)[0]
            self._state['status'][recep] = 'turned on'
        elif 'clean' in action:
            object, recep = re.findall('clean (.+) with (.+)', action)[0]
            self._state['status'][object] = 'clean'
        elif 'heat' in action:
            object, recep = re.findall('heat (.+) with (.+)', action)[0]
            self._state['status'][object] = 'heated'
        elif 'cool' in action:
            object, recep = re.findall('cool (.+) with (.+)', action)[0]
            self._state['status'][object] = 'cool'
        else:
            raise NotImplementedError

        # new_state = "You arrived at {}.\n".format(self._state['location'])
        # new_state += "visited receptacles:\n{}\n{}\n".format(
        #     '\n'.join(['{} is {}.'.format(key, status) for key, status in self._state['status'].items()]),
        #     '\n'.join(['in {}, you see {}.'.format(key, 'nothing' if len(objects) == 0 else ', '.join(objects)) for key, objects in self._state['contain'].items()])
        # )
        
        new_state = 'After a few steps, here is what you have seen, where you are, and what you are carrying.\n'

        infos = []
        new_state += "You are currently in/at {}.\n".format(self._state['location'])

        locations = [a[6:] for a in avail_actions if a.startswith('go to') and a[6:] not in self._state['status']]
        if len(locations) > 0:
            new_state += "You have been to {}.\n".format(', '.join(self.visited_locations))
            new_state += "You have not been to {}.\n".format(', '.join(locations))
        else:
            new_state += "You have been to all locations.\n"

        new_state += "You also know that:\n"
        for idx, (key, status) in enumerate(self._state['status'].items()):
            if key in self._state['contain']:
                string = "- {} is {} and there is {} in {}".format(key, status, 'nothing' if len(self._state['contain'][key]) == 0 else ', '.join(self._state['contain'][key]), key)
            else:
                string = "- {} is {}".format(key, status)
                
            string += ".\n"
            infos.append(string)
        infos = sorted(infos)
        new_state += ''.join(infos)

        new_state += "You are taking {} with you.\n".format('nothing' if len(self._state['inventory']) == 0 else self._state['inventory'][0])

        self.state = new_state


    def process_ob(self, ob, action):
        if ob.startswith('You arrive at loc '):
            ob = ob[ob.find('. ')+2:]
        if 'Next to it' in ob:
            ob = ob[:ob.find('. ') + 1]
        if 'Nothing happens' in ob:
            ob = 'Action failed. '
            if 'feedback' in self.method:
                ob += self.error_msg(action)
        return ob


    def formalize(self, action):
#         message = """Transform an action using following rules:
# go_to(a, b) -> go to a b
# take(a, b, c, d) -> take a b from c d
# put(a, b, c, d) -> put a b in/on c d
# open(a, b) -> open a b
# close(a, b) -> close a b
# use(a, b) -> use a b
# clean(a, b, c, d) -> clean a b with c d
# heat(a, b, c, d) -> heat a b with c d
# cool(a, b, c, d) -> cool a b with c d

# {} -> """.format(action)
        message = """Valid action formats are as follows:
go to "recep"
take "object" from "recep"
put "object" in/on "recep"
open "recep"
close "recep"
use "recep"
clean "object" with "recep"
heat "object" with "recep"
cool "object" with "recep"
The "object" and "recep" should be replaced with real names and indices, e.g. "apple 1" and "desk 1".

Formalize the following action strictly into the above valid action formats. If there are multiple actions, formalize the first one.

Action: I want to take laptop 1 from desk 2 and go to the drawer 1 and open drawer 1.
Formalized First Action: take laptop 1 from desk 2

Action: (1) go to desk 1
Formalized First Action: go to desk 1

Action: put heated cup 1 in/on desk 1
Formalized First Action: put cup 1 in/on desk 1

Action: {}
Formalized First Action:
""".format(action)
        message = {'role': 'user', 'content': message}
        action = self.call_openai_api([message], stop='\n')
        return action


    def same_action(self, string1, string2):
        diff = difflib.ndiff(string1, string2)
        different_characters = [char for char in diff if char[0] != ' ']
        for c in different_characters:
            if c[2:].strip(' ' + punctuation) != '':
                return False
        return True
    
    def contain_action(self, string1, string2):
        dist = editdistance.distance(string1, string2)
        return dist <= len(string1) - len(string2)
    
    def run(self, envs, strategy=None, is_test=False, verbose=False, react=False, return_history=False):
        if react:
            folder = './baselines/ReAct/prompts/'
            prompt_file = 'alfworld.json'
            with open(folder + prompt_file, 'r') as f:
                react_prompt = json.load(f)

        system_prompt = {'role': 'system', 'content': 'You are an excellent game player. Remember the game rule may differ from real world.'}

        history = ''
        results = []
        job_descs = []
        summaries = []
        flawed_actions = []
        flawed_plans = []
        for env_idx in tqdm(range(len(envs))):
            while True:
                try:
                    self.messages = [system_prompt]
                    self.visited_locations = set()

                    env = envs[env_idx]
                    ob, info = env.reset()
                    ob = '\n'.join(ob[0].split('\n\n')[1:]).replace('task', 'job')

                    plan_step = "None"

                    if strategy is not None:
                        gamma = "\nGame Plan:\n{}".format(strategy)
                        if 'step' in self.method: plan_step = strategy.split('\n')[0]
                    elif react:
                        prefix = REACT_PREFIXES[self.subtask]
                        icl_examples = react_prompt[f'react_{prefix}_1'] + react_prompt[f'react_{prefix}_0']
                        gamma = '\n\nHere are some examples.\n' + icl_examples.replace('> ', '')
                    else:
                        gamma = ''

                    self.init_state(ob, info['admissible_commands'][0])
                    job_descs.append('You job is to ' + self.objective)

                    past_actions = []

                    instance_prompt = "Game Description:\n{}{}\n\nGame Starts:\nGame Objective: {}.\n{}\n".format(self.task_description(), gamma, self.objective, self.state)
                    if 'step' in self.method and strategy is not None:
                        thought_prompt = "Identify which step of plan you are at. Show your thought about the one next action. Your thought should be faithful to the plan step."
                    else:
                        thought_prompt = "Show your thought about the next action."
                    action_prompt = "Action:"

                    init_msg = instance_prompt + thought_prompt
                    history += pretty_print('User', init_msg, verbose)
                    thought = self.call_lm(init_msg)
                    history += pretty_print('Machine', thought, verbose)

                    cnt = 0
                    success = False
                    while cnt < self.max_iter_per_instance:

                        history += pretty_print('User', 'Action {}:'.format(cnt), verbose)
                        action = self.call_lm(action_prompt, stop='\n', max_token=50)

                        if 'quit' in action.lower():
                            history += pretty_print('', 'quit', verbose)
                            self.messages.append({'role': 'assistant', 'content': 'quit'})
                            break

                        _action = self.formalize(action).lower() if 'react' not in self.method else action.lower()
                        
                        # if 'feedback' not in self.method:
                        #     action_emb = self.emb.encode([_action], convert_to_tensor=True)
                        avail_actions = []
                        formalized_action = None
                        for a in info['admissible_commands'][0]:
                            if 'inventory' not in a and 'look' not in a and 'examine' not in a:
                                avail_actions.append(a)
                                # if self.contain_action(a, _action):
                                #     formalized_action = a
                                #     break
                        # if formalized_action is None:
                        #     formalized_action = random.choice(avail_actions)
                        formalized_action = _action
                        if formalized_action.startswith('put'):
                            formalized_action = re.sub(r'put (.+) (in|on) (.+)', r'put \1 in/on \3', formalized_action)

                        if self.random_explore > 0:
                            if np.random.random() < self.random_explore:
                                formalized_action = random.choice(avail_actions)

                        history += pretty_print('Machine', "{} (v1. {}) (orig. {})".format(formalized_action, _action, action), verbose)

                        observation, reward, done, info = env.step([formalized_action])
                        observation = self.process_ob(observation[0], formalized_action)


                        cnt += 1
                        history += pretty_print('User', 'Observation {}:\n{}'.format(cnt, observation), verbose)

                        if done[0]:
                            success = True
                            break
                        
                        if 'Action failed' not in observation:
                            past_actions.append(formalized_action)
                            # reachable_locations = []
                            # for c in info['admissible_commands'][0]:
                            #     if c.startswith('go to'):
                            #         reachable_locations.append(c[5:].strip())
                            self.update_state(formalized_action, observation, info['admissible_commands'][0])
                            # self.messages.append({'role': 'user', 'content': 'State {}:\n{}\n'.format(cnt, self.state)})

                        full_thought_prompt = 'Observation: {}\n'.format(observation) + thought_prompt
                        history += pretty_print('User', thought_prompt, verbose)
                        thought = self.call_lm(full_thought_prompt, stop='\n', max_token=100)
                        history += pretty_print('Machine', thought, verbose)
                    
                    results.append(success)
                    if not is_test:
                        if not success:
                            summary_msg = 'Job failed. Maximum number of steps reached.'
                        else:
                            summary_msg = 'Job succeeded.'
                        summary_msg += ' Summarize the interaction history in steps.'

                        history += pretty_print('Human', summary_msg, verbose)
                        summary = self.call_lm(summary_msg, stop=None, lm='gpt-4-0314', top_p=0.0)
                        history += pretty_print('Machine', summary, verbose)
                        summaries.append(summary)

                        if 'no_reflection' not in self.method:
                            failed_action_msg = 'Identify all flawed parts of the plan/action. Remember in this game things are not like real world. The system message is always correct and the game plan/action may have flaws.'
                            history += pretty_print('Human', failed_action_msg, verbose)
                            failed_action = self.call_lm(failed_action_msg, stop=None, lm='gpt-4-0314', top_p=0.0)
                            history += pretty_print('Machine', failed_action, verbose)
                            flawed_actions.append(failed_action)

                            if strategy is not None:
                                suggest_rev_msg = 'Suggest revision to the current flawed part of the plan. Only the flawed part.'
                                history += pretty_print('Human', suggest_rev_msg, verbose)
                                suggest_rev = self.call_lm(suggest_rev_msg, stop=None, lm='gpt-4-0314', top_p=0.0)
                                history += pretty_print('Machine', suggest_rev, verbose)

                                flawed_plans.append(suggest_rev)

                    
                    print('\n===Cost: {}===\n'.format(self.price()))
                    break
                except Exception as e:
                    print(e)
                    if e is openai.error.InvalidRequestError:
                        results.append(False)
                        break
                    print('Error occurred, retrying...')
                    continue
        
        if is_test:
            to_return = results
        else:
            self.messages = []

            final_msg = 'Game Description:\n' + self.task_description() + '\n\n'
            final_msg += 'Current Game Plan:\n{}\n\n'.format(strategy)

            final_msg += '=' * 10 + 'Game Experiences Begin' + '=' * 10 + '\n\n'

            for env_idx in range(len(envs)):
                final_msg += 'Job {}:\n{}\nResult of Job {}:\n{}\nSummary of Job {}:\n{}\nFlawed Actions of Job {}:\n{}\n'.format(
                    env_idx, job_descs[env_idx], 
                    env_idx, 'Succeeded' if results[env_idx] else 'Failed',
                    env_idx, summaries[env_idx],
                    env_idx, flawed_actions[env_idx] if 'no_reflection' not in self.method else 'N/A',
                )
                if strategy is not None:
                    final_msg += 'Suggested Revision of Plan from Job {}:\n{}\n'.format(env_idx, flawed_plans[env_idx] if 'no_reflection' not in self.method else 'N/A')

            final_msg += '=' * 10 + 'Game Experiences End' + '=' * 10 + '\n\n'
            
            final_msg += 'Based on the above {} experiences of the game, rewrite the current game plan. Pay more attention to summary of successful jobs, and flawed actions and suggested revision of all jobs. The plan should not be specific to one job objective but generalizable to all job objectives. The actions in the plan should also be in the form as in game description. \n\nNew Game Plan:'.format(len(envs))

            history += pretty_print('Human', final_msg, verbose)
            new_strategy = self.call_lm(final_msg, stop=None, lm='gpt-4-0314', top_p=0.0)
            history += pretty_print('Machine', new_strategy, verbose)

            # final_msg += 'Refine the new game plan  \n\nNew Game Plan:'.format(len(envs))

            # history += pretty_print('Human', final_msg, verbose)
            # new_strategy = self.call_lm(final_msg, stop=None, lm='gpt-4-0314', top_p=0.0)
            # history += pretty_print('Machine', new_strategy, verbose)

            # final_msg += 'Based on the above {} experiences of the task, rewrite the current task plan in python style pseudocode using only action described in task description. Pay more attention to successful jobs if there is any. The plan should not specific to one job objective but generalizable to all job objectives. \n\nNew Task Plan:'.format(len(envs))

            # history += pretty_print('Human', final_msg, verbose)
            # new_strategy = self.call_lm(final_msg, stop=None, lm='gpt-4-0314', top_p=0.0)
            # history += pretty_print('Machine', new_strategy, verbose)

            to_return = new_strategy

        if return_history:
            to_return = to_return, history

        if not is_test:
            wandb.log({"train_succ": sum(results) / len(results)})
        
        return to_return

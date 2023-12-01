import os
import time

import openai
import json
from typing import Union, Tuple
from set_env import set_env_vars
from llm_planning.game_classes.pddl_planning_game import PDDLPlanningGame
from llm_planning.game_classes.pddl_planning_game_thoughts import PDDLGameThoughts
from llm_planning.game_classes.pddl_planning_game_planbench import PDDLGamePlanBench
from llm_planning.game_classes.pddl_planning_game_planbench_thoughts import PDDLGamePlanbenchThoughts
from utils.paths import *

set_env_vars()
openai.api_key = os.environ['OPENAI_API_KEY']


def play_games(config, few_shot_path, game_class):

    for task_num in config["task_nums"]:
        attempt = 0
        while attempt < 5:
            instance_config = config.copy()
            run_config = instance_config.pop('run_config')

            game = create_game(task_num=task_num, instance_config=instance_config, few_shot_path=few_shot_path,
                               game_class=game_class)

            try:
                game.run_instructions_all(**run_config)
                break

            except openai.error.ServiceUnavailableError as e:
                print('Warning: Server was unavailable. Will try again in a few seconds')
                with open(game.log, 'a') as log:
                    json.dump({'Failed': True, 'Error_type': 'openai.error.ServiceUnavailableError', 'Error_message': str(e)}, log)
                    log.write('\n')
                time.sleep(10)
                attempt += 1
                continue

            except openai.error.Timeout as e:
                print('Warning: Timout error. Will try again in a few seconds')
                with open(game.log, 'a') as log:
                    json.dump({'Failed': True, 'Error_type': 'openai.error.Timeout', 'Error_message': str(e)}, log)
                    log.write('\n')
                time.sleep(10)
                attempt += 1
                continue

            except openai.error.InvalidRequestError as e:
                print('Warning: Invalid Request. Will skip instance and continue with next one.')

                with open(game.log, 'a') as log:
                    json.dump({'Failed': True, 'Error_type': 'openai.error.InvalidRequestError', 'Error_message': str(e)}, log)
                    log.write('\n')
                time.sleep(10)
                break

            except openai.error.RateLimitError as e:
                print('Warning: RateLimitError. Will try again in a few seconds')

                with open(game.log, 'a') as log:
                    json.dump({'Failed': True, 'Error_type': 'openai.error.InvalidRequestError', 'Error_message': str(e)}, log)
                time.sleep(40)
                attempt += 1
                continue


def create_game(task_num, instance_config, few_shot_path, game_class) -> PDDLPlanningGame:

    instance_config = create_instance_game_config(task_num=task_num,
                                                  instance_config=instance_config,
                                                  few_shot_path=few_shot_path)

    if 'domain_dir' in instance_config:
        instance_config.pop('domain_dir')
    if 'planning_approach' in instance_config:
        instance_config.pop('planning_approach')
    if 'thoughts' in instance_config:
        instance_config.pop('thoughts')
    if 'encoding_type' in instance_config:
        instance_config.pop('encoding_type')
    game = game_class(**instance_config)

    return game

def create_instance_game_config(task_num, instance_config, few_shot_path):

    if not few_shot_path is None:
        if os.path.isfile(few_shot_path):
            few_shot_file = few_shot_path
        else:
            few_shot_file = os.path.join(few_shot_path, f'planning_examples_instance-{task_num - 1}.json')
        instance_config['llm_config']['plan']['examples_file'] = few_shot_file
        print(f'{task_num}: {few_shot_file}')

    instance_dir = instance_config['instance_dir']
    instance_file = os.path.join(instance_dir, f'instance-{task_num}.pddl')

    del instance_config['task_nums']
    del instance_config['instance_dir']
    instance_config['task_num'] = task_num
    instance_config['instance_file'] = instance_file

    return instance_config


def get_prompts(config, few_shot_path, game_class):

    task_num = config["task_nums"][0]
    instance_config = config.copy()
    instance_config.pop('run_config')

    game = create_game(task_num=task_num, instance_config=instance_config, few_shot_path=few_shot_path, game_class=game_class)

    return game.get_plan_prompt(), game.get_translate_prompt(), game.get_description_current_state()


def show_prompt(config, few_shot_path, game_class):

    plan_prompt, translate_prompt, initial_state = get_prompts(config, few_shot_path, game_class)

    print('Planning Prompt:')
    print(plan_prompt)
    print('Translation Prompt:')
    print(translate_prompt)
    print('Initial State:')
    print(initial_state)


def get_game_class(thoughts: bool, planbench: bool):

    if thoughts and planbench:
        game_class = PDDLGamePlanbenchThoughts
    elif thoughts:
        game_class = PDDLGameThoughts
    elif planbench:
        game_class = PDDLGamePlanBench
    else:
        game_class = PDDLPlanningGame

    return game_class


def set_up_configurations(config_file:str, few_shot_id: Union[int, None]) -> Tuple[dict, Union[str, None]]:

    with open(config_file, 'r') as cf:
        config = json.load(cf)

    # Fill default values for the file paths based on the specified domain directory
    domain_dir = config['domain_dir']
    if config.get('domain_file', '') == '':
        assert domain_dir
        config['domain_file'] = os.path.join(domain_dir, 'domain.pddl')
    if config.get('domain_nl_file', '') == '':
        assert domain_dir
        config['domain_nl_file'] = os.path.join(domain_dir, 'domain_description.json')
    if config.get('instance_dir', '') == '':
        assert domain_dir
        config['instance_dir'] = os.path.join(domain_dir, INST_FOLDER)

    # If no task IDs provided, select all files in the instance_dir
    if not config.get('task_nums', []):
        config['task_nums'] = []
        for file in os.listdir(config['instance_dir']):
            if os.path.isfile(file):
                task_num = file.replace('instance-', '').replace('.pddl', '')
                config['task_nums'].append(task_num)
        config['task_nums'].sort()
        config['task_nums'].pop(0)

    # Select appropriate few shot example files
    approach_type = config['planning_approach']

    if not few_shot_id:
        few_shot_path = None
    else:
        few_shot_dir_path = get_few_shot_dir(planning_approach=approach_type, domain_data_dir=domain_dir)
        few_shot_path = get_few_shot_ex_file(few_shot_dir=few_shot_dir_path, instance_id=few_shot_id, approach=approach_type)

    return config, few_shot_path


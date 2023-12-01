import argparse
from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import sys
import time

# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from calvin_env.envs.play_table_env import get_env

import openai
import cohere
import re # regex for locating the subtask items in the output of LLM

logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 1000


class CALVINRobotManager():
    def __init__(self, model, env, eval_log_dir=None, visualize=False, easy_mode=True, cohere_api_key=None):
        """
        Interface between CALVIN pretrained model + CALVIN env and LLM
        
        Currently set to an easy task (Place blue cube in drawer)
        
        # TODO:
            - Refactor so can do multiple different tasks
            - Env description oracle
            - Allow LLM to choose from full range of actions
        """
        assert easy_mode
        
        self.model = model # pretrained hulc actor/model
        self.env = env
        self.visualize = visualize
        self.cohere_api_key = cohere_api_key

        self.conf_dir = Path(__file__).absolute().parents[2] / "conf"
        task_cfg = OmegaConf.load(self.conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
        self.eval_log_dir = get_log_dir(eval_log_dir)
        
        # Used to check whther task is succesful
        self.task_oracle = hydra.utils.instantiate(task_cfg)
        
        # A dict where keys are the actions the robot can take, and items are the descriptions of the action
        self.subtasks_dict = OmegaConf.load(self.conf_dir / "annotations/new_playtable_validation.yaml")
        
        for k,v in self.subtasks_dict.items():
            print(f"{k}: {v}")
        
        # A list of valid sequences. Each is tuple of (initial_state, eval_sequence)
        # eval_sequence is a list of actions
        self.eval_sequences = get_sequences(NUM_SEQUENCES)
        # for initial_state, eval_sequence in self.eval_sequences:
            
        self.env_is_active = False

        self.frames = []
        

    def reset_env(self):
        """
        Use to initialize the environment.

        Returns
        -------
        Strings describing the envioronment, task, and available actions (info to feed to LLM)

        """
        # get intial state info
        initial_state = self.get_initial_state()
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        # reset env
        self.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        self.env_is_active = True
        return self.get_env_description(), self.get_task_description(), self.get_available_actions_description()
    
    
    def rollout(self, subtask, override_subtask=False):
        """
        Attempt to perform one action/subtask in the environment.
        Returns True/False for success/failure.

        Parameters
        ----------
        subtask : 
            A string describing the action/subtask to be completed by the robot.
            Must be valid (i.e. must be one of the keys in self.subtasks_dict).

        Returns
        -------
        True/False for success/failure.

        """
        
        assert self.env_is_active
        
        # custom subtask
        if override_subtask:
            subtask = self.user_select_single_subtask()
        # get lang annotation for subtask
        lang_annotation = self.subtasks_dict[subtask][0]
        assert subtask in self.subtasks_dict.keys()
        
        obs = self.env.get_obs()
        self.model.reset()
        start_info = self.env.get_info()

        
        if self.visualize:
            print(f"\nAttempting {subtask}...")
            time.sleep(0.5)
            img = self.env.render(mode="rgb_array")
            join_vis_lang(img, "awaiting subtask...")
            input("Press [Enter] to begin rollout...")
            self.frames.append(img)

        for step in range(EP_LEN):
            action = self.model.step(obs, lang_annotation)
            obs, _, _, current_info = self.env.step(action)
            
            if self.visualize:
                img = self.env.render(mode="rgb_array")
                join_vis_lang(img, lang_annotation)
                # time.sleep(0.1)
            
            # check if current step solves the task
            current_task_info = self.task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                if self.visualize:
                    print(colored("success", "green"), end=" ")
                return True
        if self.visualize:
            print(colored("fail", "red"), end=" ")
        return False


    def get_initial_state(self):
        # Just initialize to same state each time.
        initial_state = {
                'led': 1, 'lightbulb': 1, 'slider': 'left', 'drawer': 'closed', 'red_block': 'slider_left', \
                'blue_block': 'table', 'pink_block': 'slider_right', 'grasped': 0
                }
        return initial_state
        
    
    """
    Getting env + task string descriptions
    
    """
    
    def get_env_description(self):
        # Only describing info relevant to simple 'place blue cube in drawer' task.
        return "There is a robot in front of a table. The table has a closed drawer. There is a blue block placed on the table."
        
    def get_task_description(self):
        return "The robot's goal is to put the blue block into the drawer."
        
    def get_available_actions_description(self):
        """
        The actions available to the LLM.
        Only providing actions relevant to the task.
        
        """
        valid_actions = ['lift_blue_block_table', 'place_in_drawer', 'open_drawer', 'close_drawer']
        actions_str = "The actions the robot can take include:"
        for valid_act in valid_actions:
            actions_str += f"\n- Action: '{valid_act}', {self.subtasks_dict[valid_act][0]}"
        return actions_str
        
    def get_current_env_description(self):
        scene_info = self.env.get_info()["scene_info"]
        print(scene_info.keys())

        position_dict = {
            "cabinet": 6,
            "drawer": 3,
            "table": -1
        }

        block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
        block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])

        object_states = []

        for key in scene_info.keys():
            # print("="*20, f" {key} ", "="*20)
            # for k in scene_info[key].keys():
            #     if key == "movable_objects":
            #         print(k, scene_info[key][k]["uid"])
            #         print(scene_info[key][k])
            #     else:
            #         print(k, scene_info[key][k])

            # get the current state of moveable objects according to contacts
            if key == "movable_objects":
                for obj in scene_info[key].keys():
                    contacts = scene_info[key][obj]["contacts"][0]
                    links = contacts[3:5]

                    tmp = obj.split("_")
                    obj_string = f"The {tmp[1]} {tmp[0]}"

                    if links[0] == links[1] == -1:
                        object_states.append(f"{obj_string}. state: on the table.")

                    if links[-1] == 6:
                        current_pos = np.asarray(scene_info[key][obj]["current_pos"])
                        left_dist = np.linalg.norm(current_pos - block_slider_left)
                        right_dist = np.linalg.norm(current_pos - block_slider_right)
                        if left_dist < right_dist:
                            object_states.append(f"{obj_string}. state: inside the cabinet, on the left side.")
                        else:
                            object_states.append(f"{obj_string}. state: inside the cabinet, on the right side.")
                    if links[-1] == 3:
                        object_states.append(f"{obj_string}. state: in the drawer.")
                    if links[-1] == 10:
                        object_states.append(f"{obj_string}. state: grasped by the robot.")
            if key == "doors":
                current_state_slide = scene_info[key]["base__slide"]["current_state"]
                if current_state_slide == 0.28:
                    object_states.append("The sliding door. state: on the left side of the cabinet.")
                else:
                    object_states.append("The sliding door. state: on the right side of the cabinet.")
                
                current_state_drawer = scene_info[key]["base__drawer"]["current_state"]
                if current_state_drawer == 0.0:
                    object_states.append("The drawer. state: closed.")
                else:
                    object_states.append("The drawer. state: opened.")
            if key == "lights":
                current_state_bulb = scene_info[key]["lightbulb"]["logical_state"]
                if current_state_bulb == 1:
                    object_states.append("The lightbulb. state: on.")
                else:
                    object_states.append("The lightbulb. state: off.")
                

                current_state_bulb = scene_info[key]["led"]["logical_state"]
                if current_state_bulb == 1:
                    object_states.append("The led. state: on.")
                else:
                    object_states.append("The led. state: off.")
        object_states_prompt = "\n\t".join(object_states)

        print(object_states_prompt)

        return object_states_prompt



    def print_available_subtasks(self):
        print()
        for i, key in enumerate(self.subtasks_dict.keys()):
            print(f"[i] {key}: {self.subtasks_dict[key]}")
        print()
        
    def generate_llm_prompt(self, env_str, task_str, actions_str):
        prompt = f"Consider the following environment: \n'{env_str}'\n"
        prompt += task_str + "\n"
        prompt += actions_str
        prompt += "List the sequence of actions the robot should take to complete the task."
        prompt += "\n"
        prompt += "\nThe robot should take the following sequence of actions to complete the task:"
        
        return prompt
    
    
    """
    Selecting plans
    
    """
    
    def select_subtask_sequence(self, env_str, task_str, actions_str, planner='ground-truth'):
        prompt = self.generate_llm_prompt(env_str, task_str, actions_str)
        if planner == 'truth':
            return self.get_groundtruth_sequence()
        elif planner == 'user':
            return self.user_select_sequence()
        elif planner == 'openai':
            return self.openai_select_subtask(task_str)
        elif planner == 'cohere':
            return self.cohere_select_subtask(prompt)
        else:
            assert False
        
    def get_groundtruth_sequence(self):
        return ["turn_off_led", "open_drawer", "move_slider_right", "lift_red_block_slider", "place_in_drawer", "close_drawer", "turn_off_lightbulb"]
        # return ['turn_off_led', 'open_drawer', 'move_slider_right', 'lift_red_block_slider', 'place_in_drawer', 'close_drawer', 'move_slider_left', 'lift_pink_block_slider', 'stack_block', 'turn_off_lightbulb']
     
    def user_select_sequence(self):
        n = int(input("Select number of subtasks to perform [i]: "))
        assert n > 0 and n <= 4
        steps = []
        for i in range(n):
            print(f"Selecting subtask {[i]}/{[n]}")
            chosen_step = self.user_select_single_subtask()
            steps.append(chosen_step)
        return steps
        
    def user_select_single_subtask(self):
        print()
        for i, key in enumerate(self.subtasks_dict.keys()):
            print(f"[{i}] {key}")
        print()
        selected_i = int(input("Select subtask to perform [i]: "))
        subtask = list(self.subtasks_dict.keys())[selected_i]
        print(f"\nYou selected {subtask} ({self.subtasks_dict[subtask]})")
        return subtask
        
    def openai_select_subtask(self, task):
        object_state = self.get_current_env_description()

        print("\nUse OpenAI planner")
        openai.api_key = "your-openai-api-key"
        engine = openai.Engine.list()

        API_identity_prompt= """
        I want you to act like a single-arm franka robot.
        """

        enviroment_prompt= f"""
        There is a table in front of you. 
        There is a drawer below the desktop. 
        There is a black button on the desktop. 
        On the top side of the desktop, there is a shelf, a led and a lightbulb are placed on the top of the shelf. 
        The lightbulb is placed on the left side while the yellow bulb is placed on the right side. 
        The lightbulb is controlled by the black botton.
        The shelf itself, is compose of two parts. 
        The left part of the shelf is a cabinet with a sliding door.  
        The right part of the shelf is a switch which can be pressed down to control lightbulb.
        """

        object_state_prompt = f"""
        Now here are the current states of the objects:
        {object_state}

        """
        task_requirement_prompt="""
        Here is you skill pool and the descriptions of each skill:
        1. open_drawer. pull the handle and open the drawer. this task can only be executed without an object grasped in hand.
        2. lift_blue_block_table. grasp and lift the blue block and grasp it in hand.
        3. place_in_drawer.  place the grasped object in the drawer.
        4. close_drawer. push the handle and close the drawer.
        5. turn_on_led. press the button to turn on the led light.
        6. turn_off_led. press the button to turn off the led light.
        7. turn_on_lightbulb. use the switch to turn on the light bulb.
        8. turn_off_lightbulb. use the switch to turn off the light bulb.
        9. push_into_drawer. push the block and make it falls into the drawer.
        10. move_slider_left. push the sliding door of the cabinet to the left side.
        11. move_slider_right. push the sliding door of the cabinet to the right side.
        12. lift_red_block_slider. lift the red block from the sliding cabinet and grasp it in hand.
        13. lift_blue_block_slider. lift the blue block from the sliding cabinet and grasp it in hand.
        14. lift_pink_block_slider. lift the pink block from the sliding cabinet and grasp it in hand.
        15. stack_block. stack the grasped block on another block.

        Here are several rules that you must follow:
        1. To place an object in the closed drawer, you must open the drawer first.
        2. You can not open the drawer after executing a picking action, for example: lift_blue_block_table, lift_red_block_slider, lift_blue_block_slider, lift_pink_block_slider.
        3. The object on left side of cabinet can only be picked when the sliding door on the right side.
        4. The object on right side of cabinet can only be picked when the sliding door on the left side.
        5. The sliding door can only be moved to left side when it is on the right side.
        6. The sliding door can only be moved to right side when it is on the left side.
        7. Each sub-tasks must can be completed by a single skill from the skill pool.
        8. All sub-tasks should be listed in a python-style list, here is an example ["[ACT]", "[ACT]", "[ACT]", "[ACT]", ...], [ACT]
        is a sub-task
        9. You just need to list all sub-tasks in a python-style list. No other words are needed.
        """

        task_prompt = f"""
        Now, there is a command: "{task}". Please help me decompose this task into a sequence of sub-tasks. you must plan the sub-tasks based on the current states of the objects.
        """

        prompt = API_identity_prompt + enviroment_prompt + object_state_prompt + task_requirement_prompt + task_prompt

        print("\nPrompt: \n", prompt)

        # Use OpenAI's language model to generate a list of steps for achieving the goal
        ####################################
        # API for calling text-davinci-003 #
        ####################################
        # response = openai.Completion.create(
        #     engine="text-davinci-003", 
        #     # # engine="text-curie-001", 
        #     prompt=prompt,
        #     max_tokens=128,
        #     temperature=0.7
        # )


        ####################################
        #        API for calling GPT-4     #
        ####################################
        response = openai.ChatCompletion.create(
            model="gpt-4", 
            # # engine="text-curie-001", 
            messages=[
                {"role": "system", "content": "You are a helpful robot planner."},
                {"role": "user", "content": f"{prompt}"},
            ]
        )
        print(response)

        ######################################################
        # Needs to be modified to suit output style of GPT-4 #
        ######################################################
        # text = response.choices[0].text
        # print("\nOpenAI response: ", text)
        # text = text.replace("[", "").replace("]", "").replace("\n", "").replace(" ", "").strip()
        # text = text.replace(".", "").replace(":", "")

        # steps = text.split(",")

        # actions = []

        # for step in steps:
        #     actions.append(str(step[1:-1]))
        
        # print(actions)

        
        # # Return the list of generated steps
        # return actions
    
    def cohere_select_subtask(self, prompt):
        co = cohere.Client(self.cohere_api_key)

        # Use Cohere's language model to generate a list of steps for achieving the goal
        response = co.generate(  
            model='command-xlarge',  
            prompt = prompt,  
            max_tokens=40,  
            temperature=0.3,  
            stop_sequences=None)
        print("Cohere response: ", response)

        steps = response.generations[0].text
        # Extract the generated steps from the response
        steps = steps.strip().split("\n")
        # Remove the policy choices from the generated steps
        steps = [step.split(". ", 1)[1].strip() for step in steps]

        # Return the list of generated steps
        return steps

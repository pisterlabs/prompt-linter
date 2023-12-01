import copy
import json
import os
import time
from typing import Dict

import utils as U
from env.bridge import FoyagerEnv

from agents import ActionAgent
from agents import CriticAgent
from agents import CurriculumAgent
from agents import SkillManager
import anthropic


# TODO: remove event memory
class Foyager:
    def __init__(
        self,
        server_ip = "127.0.0.1",
        rcon_port: int = None,
        rcon_password: str =  None,
        openai_api_key: str = None,
        anthropic_api_key:str=None,
        env_wait_ticks: int = 20,
        env_request_timeout: int = 600,
        max_iterations: int = 160,
        reset_placed_if_failed: bool = False,
        action_agent_model_name: str = "gpt-3.5-turbo",
        action_agent_temperature: int = 0,
        action_agent_task_max_retries: int = 4,
        action_agent_show_chat_log: bool = True,
        action_agent_show_execution_error: bool = True,
        curriculum_agent_model_name: str = "gpt-3.5-turbo",
        curriculum_agent_temperature: int = 0,
        curriculum_agent_qa_model_name: str = "gpt-3.5-turbo",
        curriculum_agent_qa_temperature: int = 0,
        curriculum_agent_warm_up: Dict[str, int] = None,
        curriculum_agent_core_inventory_items: str = r".*_log|.*_planks|stick|crafting_table|furnace"
        r"|cobblestone|dirt|coal|.*_pickaxe|.*_sword|.*_axe",
        curriculum_agent_mode: str = "auto",
        critic_agent_model_name: str = "gpt-3.5-turbo",
        critic_agent_temperature: int = 0,
        critic_agent_mode: str = "auto",
        skill_manager_model_name: str = "gpt-3.5-turbo",
        skill_manager_temperature: int = 0,
        skill_manager_retrieval_top_k: int = 5,
        openai_api_request_timeout: int = 240,
        ckpt_dir: str = "ckpt",
        resume: bool = False,
    ):
        """
        The main class for Voyager.
        Action agent is the iterative prompting mechanism in paper.
        Curriculum agent is the automatic curriculum in paper.
        Critic agent is the self-verification in paper.
        Skill manager is the skill library in paper.
        :param rcon_port: factorio rcon port
        :param openai_api_key: openai api key
        :param env_wait_ticks: how many ticks at the end each step will wait, if you found some chat log missing,
        you should increase this value
        :param env_request_timeout: how many seconds to wait for each step, if the code execution exceeds this time,
        python side will terminate the connection and need to be resumed
        :param reset_placed_if_failed: whether to reset placed blocks if failed, useful for building task
        :param action_agent_model_name: action agent model name
        :param action_agent_temperature: action agent temperature
        :param action_agent_task_max_retries: how many times to retry if failed
        :param curriculum_agent_model_name: curriculum agent model name
        :param curriculum_agent_temperature: curriculum agent temperature
        :param curriculum_agent_qa_model_name: curriculum agent qa model name
        :param curriculum_agent_qa_temperature: curriculum agent qa temperature
        :param curriculum_agent_warm_up: info will show in curriculum human message
        if completed task larger than the value in dict, available keys are:
        {
            "context": int,
            "biome": int,
            "time": int,
            "other_blocks": int,
            "nearby_entities": int,
            "health": int,
            "hunger": int,
            "position": int,
            "equipment": int,
            "chests": int,
            "optional_inventory_items": int,
        }
        :param curriculum_agent_core_inventory_items: only show these items in inventory before optional_inventory_items
        reached in warm up
        :param curriculum_agent_mode: "auto" for automatic curriculum, "manual" for human curriculum
        :param critic_agent_model_name: critic agent model name
        :param critic_agent_temperature: critic agent temperature
        :param critic_agent_mode: "auto" for automatic critic ,"manual" for human critic
        :param skill_manager_model_name: skill manager model name
        :param skill_manager_temperature: skill manager temperature
        :param skill_manager_retrieval_top_k: how many skills to retrieve for each task
        :param openai_api_request_timeout: how many seconds to wait for openai api
        :param ckpt_dir: checkpoint dir
        :param resume: whether to resume from checkpoint
        """
        # init env
        self.env = FoyagerEnv(
            server_ip=server_ip,
            rcon_port=rcon_port,
            rcon_password=rcon_password,
        )
        self.env_wait_ticks = env_wait_ticks
        self.reset_placed_if_failed = reset_placed_if_failed
        self.max_iterations = max_iterations
        self.resume = resume

        # set openai api key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.envirorn = os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

        # init agents
        self.action_agent = ActionAgent(
            model_name=action_agent_model_name,
            env=self.env,
            ckpt_dir=ckpt_dir,
            resume=resume,
            chat_log=action_agent_show_chat_log,
            execution_error=action_agent_show_execution_error,
        )
        self.action_agent_task_max_retries = action_agent_task_max_retries
        self.curriculum_agent = CurriculumAgent(
            model_name=curriculum_agent_model_name,
            temperature=curriculum_agent_temperature,
            qa_model_name=curriculum_agent_qa_model_name,
            qa_temperature=curriculum_agent_qa_temperature,
            request_timout=openai_api_request_timeout,
            ckpt_dir=ckpt_dir,
            resume=resume,
            mode=curriculum_agent_mode,
            warm_up=curriculum_agent_warm_up,
            core_inventory_items=curriculum_agent_core_inventory_items,
        )
        self.skill_manager = SkillManager(
            model_name=skill_manager_model_name,
            temperature=skill_manager_temperature,
            retrieval_top_k=skill_manager_retrieval_top_k,
            request_timout=openai_api_request_timeout,
            ckpt_dir=ckpt_dir,
            resume=resume,
        )
        self.recorder = U.EventRecorder(ckpt_dir=ckpt_dir)
        self.resume = resume

        # init variables for rollout
        self.action_agent_rollout_num_iter = -1
        self.task = None
        self.context = ""
        self.messages = None
        self.conversations = []
        self.last_events = None

    """
        The reset method resets the agent's state and the game environment to prepare for a new task. 
        It initializes the environment, sets the difficulty level, retrieves skills from the skill manager, 
        and generates system and human messages for communication.
        
        Args:
            task: idk
            context: idk
            reset_env: Whether we should hard reset the environment
            
        Returns:
            messages: idk
    """
    def reset(self, task, context="", reset_env=True, ):
        self.action_agent_rollout_num_iter = 0
        self.task = task
        self.context = context
        if reset_env:
            self.env.reset(mode='soft',
                           wait_ticks= self.env_wait_ticks,
                           refresh_entities=['resource','simple-entitiy']
                           )

        
        skills = self.skill_manager.retrieve_skills(query=self.context)
        #first observation is only does resources and simple-entities
        events = self.env.step(refresh_entities=['resource','simple-entitiy','assembling-machine','furnace'])
        print(
            f"\033[33mRender Action Agent system message with {len(skills)} control_primitives\033[0m"
        )

        system_message = self.action_agent.render_system_message(skills=skills)
        print(
            f"\023[32m****Action System message****\n{system_message.content}\023[0m"
        )
        human_message = self.action_agent.render_human_message(
            events=events, code="", task=self.task, context=context, critique=""
        )
        self.messages = [system_message, human_message]
        print(
            f"\033[32m****Action Agent human message****\n{human_message.content}\033[0m"
        )
        assert len(self.messages) == 2
        self.conversations = []
        return self.messages

    def close(self):
        self.env.close()
    
    """
    The step method performs a single step in the task execution process. 
    It generates an AI message using the action agent, processes the message, executes the code in the game environment, 
    checks task success using the critic agent, and updates the system and human messages for the next step.
        
    Args:
        None 
        
    Returns:
        messages: idk, 
        num: 0(idk), 
        bool: If the task is done
        info: idk
    """
    def step(self):
        if self.action_agent_rollout_num_iter < 0:
            raise ValueError("Agent must be reset before stepping")
        
        ai_message = self.action_agent.llm.completion(
        prompt=f"{anthropic.HUMAN_PROMPT}{self.messages[0]}\nSYSTEM MESSAGE:\n{self.messages[1]}{anthropic.AI_PROMPT}",
        stop_sequences=[],
        model="claude-v1",
        max_tokens_to_sample=1000,
        stream=False)

        print(f"\033[34m****Action Agent ai message****\n{ai_message['completion']}")
                
        self.conversations.append(
            (self.messages[0], self.messages[1], ai_message['completion'])
        )
        
        parsed_result = self.action_agent.process_ai_message(message=ai_message)
        success = False
        
        
        if isinstance(parsed_result, dict):
            code = parsed_result["program_code"]
            function_name = parsed_result["function_name"]
            events = self.env.step(
                code= code,function_name=function_name
            )
            self.recorder.record(events, self.task)
            success, critique = self.critic_agent.check_task_success(
                events=events,
                task=self.task,
                context=self.context,
                max_retries=5,
            )

        assert len(self.messages) == 2
        self.action_agent_rollout_num_iter += 1
        done = (
            self.action_agent_rollout_num_iter >= self.action_agent_task_max_retries
            or success
        )
        info = {
            "success": success,
            "conversations": self.conversations,
        }
        if success:
            assert (
                "program_code" in parsed_result and "program_name" in parsed_result
            ), "program and program_name must be returned when success"
            info["program_code"] = parsed_result["program_code"]
            info["program_name"] = parsed_result["program_name"]
        else:
            print(
                f"\033[32m****Action Agent human message****\n{self.messages[-1].content}\033[0m"
            )
        return self.messages, 0, done, info


    """
    The `rollout` method executes a complete task rollout by repeatedly calling the `step` method until the task is completed
    or a maximum number of iterations is reached. It returns the final messages, reward, completion status,
    and additional information about the rollout.
    
    Args:
        *: idk
         task: the task to complete
         context: idk
         reset_env: Whether to hard reset the environment     
        
    Returns:
        messages: idk, 
        reward: idk (presumably some type of reward), 
        context: idk
        info: idk
    """
    def rollout(self, *, task, context, reset_env=True):
        self.reset(task=task, context=context, reset_env=reset_env)
        while True:
            messages, reward, done, info = self.step()
            if done:
                break
        return messages, reward, done, info
    
    """
    The learn method is the main training loop of the Voyager system. It proposes the next task using the curriculum agent,
    performs rollouts for the proposed task, handles task completion or failure, and manages the curriculum and skill manager.
    
    Args:
        reset_env: Whether to hard reset the environment     
        
    Returns:
        {
            "completed_tasks": self.curriculum_agent.completed_tasks,
            "failed_tasks": self.curriculum_agent.failed_tasks,
            "control_primitives": self.skill_manager.skills,
        }
    """
    def learn(self, reset_env=True):
        if self.resume:
            # keep the inventory
            self.env.reset(mode='soft',
                           wait_ticks= 0,)
        else:
            # clear the inventory
            self.env.reset(mode='hard',
                refresh_entities=['resources','simple-entitiy'],
                wait_ticks=0)
            self.resume = True
        self.last_events = self.env.step(refresh_entities=['resources','simple-entitiy'])
        while True:
            if self.recorder.iteration > self.max_iterations:
                print("Iteration limit reached")
                break
            task, context = self.curriculum_agent.propose_next_task(
                events=self.last_events,
                max_retries=5,
            )
            print(
                f"\033[35mStarting task {task} for at most {self.action_agent_task_max_retries} times\033[0m"
            )

            messages, reward, done, info = self.rollout(
                    task=task,
                    context=context,
                    reset_env=reset_env,
                )
            # except Exception as e:
            #     time.sleep(3)  # wait for mineflayer to exit
            #     info = {
            #         "success": False,
            #     }
            #     # reset inventory here
            #     self.last_events = self.env.reset()
            #     # use red color background to print the error
            #     print("Your last round rollout terminated due to error:")
            #     print(f"\033[41m{e}\033[0m")
            if (
                task == "Place and deposit useless items into a chest"
                or task.startswith("Deposit useless items into the chest at")
            ):
                continue
            if info["success"]:
                print(f"\033[35mCompleted task {task}.\033[0m")
                self.skill_manager.add_skill(
                    program_name=info["program_name"],
                    program_code=info["program_code"],
                )
                self.curriculum_agent.completed_tasks.append(task)
            else:
                self.curriculum_agent.failed_tasks.append(task)
                print(
                    f"\033[35mFailed to complete task {task}. Skipping to next task.\033[0m"
                )
            # clean up tasks and dump to disk
            self.curriculum_agent.clean_up_tasks()
            print(
                f"\033[35mCompleted tasks: {', '.join(self.curriculum_agent.completed_tasks)}\033[0m"
            )
            print(
                f"\033[35mFailed tasks: {', '.join(self.curriculum_agent.failed_tasks)}\033[0m"
            )

        return {
            "completed_tasks": self.curriculum_agent.completed_tasks,
            "failed_tasks": self.curriculum_agent.failed_tasks,
            "control_primitives": self.skill_manager.skills,
        }
        
    """
    The inference method is used for inference mode, where the agent performs tasks without learning. 
    It takes a specific task and a set of sub-tasks, resets the environment, executes the sub-tasks one by one, 
    handles task completion or failure, and provides early stopping based on specified conditions.
    
    Args:
        task, 
        reset_mode="hard", 
        reset_env=True, 
        early_stop=False, 
        sub_tasks=None
    Returns:
        None
    """
    def inference(
        self, task, reset_mode="hard", reset_env=True, early_stop=False, sub_tasks=None
    ):
        self.env.reset(
            options={
                "mode": reset_mode,
                "wait_ticks": self.env_wait_ticks,
            }
        )
        self.curriculum_agent.completed_tasks = []
        self.curriculum_agent.failed_tasks = []
        self.last_events = self.env.step("")
        if not sub_tasks:
            sub_tasks = self.curriculum_agent.decompose_task(task, self.last_events)
        iter_without_new_item = 0
        last_item_history = set()
        while self.curriculum_agent.progress < len(sub_tasks):
            next_task = sub_tasks[self.curriculum_agent.progress]
            context = self.curriculum_agent.get_task_context(next_task)
            print(
                f"\033[35mStarting task {next_task} for at most {self.action_agent_task_max_retries} times\033[0m"
            )
            messages, reward, done, info = self.rollout(
                task=next_task,
                context=context,
                reset_env=reset_env,
            )
            if not self.recorder.item_history - last_item_history:
                iter_without_new_item += 1
            else:
                iter_without_new_item = 0
            last_item_history = self.recorder.item_history.copy()
            if iter_without_new_item >= 3 and early_stop:
                print("Early stop")
                break
            if info["success"]:
                print(f"\033[35mCompleted task {next_task}.\033[0m")
                self.curriculum_agent.completed_tasks.append(next_task)
            else:
                print(
                    f"\033[35mFailed to complete task {next_task}. Skipping to next task.\033[0m"
                )
                self.curriculum_agent.failed_tasks.append(next_task)

            # clean up tasks and dump to disk
            self.curriculum_agent.clean_up_tasks()
            print(
                f"\033[35mCompleted tasks: {', '.join(self.curriculum_agent.completed_tasks)}\033[0m"
            )
            print(
                f"\033[35mFailed tasks: {', '.join(self.curriculum_agent.failed_tasks)}\033[0m"
            )
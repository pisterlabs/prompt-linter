import os
import openai
# openai.api_key = os.environ['OPENAI_API_KEY']
# openai.api_key = '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'
openai.proxy = "http://127.0.0.1:7895"
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 
import json
with open('/mnt/gluster/home/zhihongyan/Project/NavGPT/data/api_key.json', 'r') as f:
    api_key = json.load(f)
api_pay_key = api_key['pay']
KEY_INDEX = 0
api_key_list = api_key['free']
import os
from tool.visual_foundation_models import VisualFoundationModels
import torch
from PIL import Image
import time
import numpy as np
BASE_LOG_DIR = os.environ['BASE_LOG_DIR']
import logging

class NavGPT():
    
    def __init__(self) -> None:
        self.history_chats = []
        self.viewpoint_summary_cache = []
        self.last_angle = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.VFM = VisualFoundationModels(self.device)
        self.step = 0
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def query_gpt3(self, messages):
        if os.environ['API_MODE'] == 'PAY':
            openai.api_key = api_pay_key
        else:
            openai.api_key = api_key_list[KEY_INDEX]
            KEY_INDEX += 1
            if KEY_INDEX == len(api_key_list):
                KEY_INDEX = 0
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=messages
            )
        response = completion.choices[0]['message']['content']
        # ['prompt_tokens', 'completion_tokens', 'total_tokens']
        token_info = completion.usage
        return response, token_info

    def _prompt2message(self, prompt):
        return [{"role": "user", "content": prompt}]
    
    def _response2message(self, response):
        return [{"role": "assistant", "content": response}]
    
    def parse_system_prompt(self, instruction):
        if os.environ['MODE'] == 'oracle':
            system_prompt = f'You are an intelligent embodied agent that follows an instruction to navigate in an indoor environment.\
Your task is to move among the static viewpoints (positions) of a pre-defined graph of the environment, \
and try to reach the target viewpoint as described by the given instructionwith the least steps.\n\
At the beginning of the navigation, you will be given an instruction of a trajectory which describes all \
observations and the action you should take at each step.During navigation, at each step, you will be at \
a specific viewpoint and receive the history of previous steps you have taken (containing your "Thought", \
"Action", "Action Input" and"Observation" after the "Begin!" sign) and the observation of current \
viewpoint (including scene descriptions, objects, and navigable directions/distances within 3 meters). \
Orientations range from -180 to 180 degrees: "0" signifies forward, "right 90" rightward, "right(or left) \
180" backward, and "left 90" leftward. \n\
You make actions by selecting navigable viewpoints to reach the destination. You are encouraged to explore \
the environment while avoiding revisiting viewpoints by comparing current navigable and previously \
visited IDs in previous "Action Input". The ultimate goal is to stop within 3 meters of the destination \
in the instruction. If destination visible but the target object is not detected within 3 meters, \
move closer. \n\
At each step, you should consider:(1) According to Current Viewpoint observation and History, have you \
reached the destination? If yes you should stop, output the \'Final Answer: Finished!\' to stop.If not \
you should continue: \n (2) Consider where you are on the trajectory and what should be the next viewpoint \
to navigatea ccording to the instruction.Use the action_maker tool, input the next navigable viewpoint ID in current observation\
to move to that location.Show your reasoning in the Thought section. \n\
Here are the descriptions of the action_maker tool: \n Can be used to move to next adjacent viewpoint. \n \
The input to this tool should be a viewpoint ID string of the next viewpoint you wish to visit. \n \
For example: \nAction: action_maker \n Action Input: "0001". \n\
Every viewpoint has a unique viewpoint ID. You are very strict to the viewpoint ID and will  \
never fabricate none xistent IDs.\
----\nRemember only output one Thought and Action once\n\
Remember only select one candidate navigable viewpoint from the current observation. viewpoint ID in \'History Observation\' is unaviable\n\
Do not select base on the history observations.\n\
Starting below, you should strictly follow this format: Instruction: an instruction of a trajectory which  \
describes all observations and the actionsshould be taken \n Initial Observation: the initial observation \
of the environment \n Thought: you should always think about what to do next and why \n \
Action: the action to take, must be one of the tools [action_maker] \n Action Input: "Viewpoint ID" \n\
----Begin!\n\
Instruction: {instruction}\n'  
        elif os.environ['MODE'] == 'normal':
           system_prompt = f'You are an intelligent embodied agent that follows an instruction to navigate in an indoor environment.\
Your task is to move among the static viewpoints (positions) of a pre-defined graph of the environment, \
and try to reach the target viewpoint as described by the given instructionwith the least steps.\n\
At the beginning of the navigation, you will be given an instruction of a trajectory which describes all \
observations and the action you should take at each step.During navigation, at each step, you will be at \
a specific viewpoint and receive the history of previous steps you have taken (containing your "Thought", \
"Action", "Action Input" and"Observation" after the "Begin!" sign) and the observation of current \
viewpoint (including scene descriptions, objects, and navigable directions/distances within 3 meters). \
Orientations range from -180 to 180 degrees: "0" signifies forward, "right 90" rightward, "right(or left) \
180" backward, and "left 90" leftward. \n\
You make actions by selecting navigable viewpoints to reach the destination. You are encouraged to explore \
the environment while avoiding revisiting viewpoints by comparing current navigable and previously \
visited IDs in previous "Action Input". The ultimate goal is to stop within 3 meters of the destination \
in the instruction. If destination visible but the target object is not detected within 3 meters, \
move closer. \n\
At each step, you should consider:(1) According to Current Viewpoint observation and History, have you \
reached the destination? If yes you should stop, output the \'Final Answer: Finished!\' to stop.If not \
you should continue: \n (2) Consider where you are on the trajectory and what should be the next viewpoint \
to navigatea ccording to the instruction.Use the action_maker tool, input the next navigable viewpoint ID in current observation\
to move to that location.Show your reasoning in the Thought section. \n\
Here are the descriptions of the action_maker tool: \n Can be used to move to next adjacent viewpoint. \n \
The input to this tool should be a viewpoint ID string of the next viewpoint you wish to visit. \n \
For example: \nAction: action_maker \n Action Input: "0001". \n\
Every viewpoint has a unique viewpoint ID. You are very strict to the viewpoint ID and will  \
never fabricate none xistent IDs.\
----\nRemember only output one Thought and Action once\n\
Remember only select one candidate navigable viewpoint from the current observation. viewpoint ID in \'History Observation\' is unaviable\n\
Do not select base on the history observations.\n\
Remeber output \'Final Answer: Finished\' when you think you arrive the target position.\n\
Please be carefully to output the \'Final Answer: Finished!\' only when you think you have finish the whole instruction but not a substep.\n\
Starting below, you should strictly follow this format: Instruction: an instruction of a trajectory which  \
describes all observations and the actionsshould be taken \n Initial Observation: the initial observation \
of the environment \n Thought: you should always think about what to do next and why \n \
Action: the action to take, must be one of the tools [action_maker] \n Action Input: "Viewpoint ID" \n\
----Begin!\n\
Instruction: {instruction}\n'                                                                     

        return system_prompt

    def parse_history_message(self, viewpoint_textual_description, thought, last_action, cur_angle=None):
        SUMMARY_PROMPT = f'Given the description of a viewpoint.\
Summarize the scene from the viewpoint in one concise sentence. Remeber just output one sentence\n\
Description:\n\
{viewpoint_textual_description}\n\
Summarization: The scene from the viewpoint is'

        msg = self._prompt2message(SUMMARY_PROMPT)
        viewpoint_summary, _ = self.query_gpt3(msg) 
                
        # cur_angle_msg = f'left {cur_angle}' if cur_angle < 180 else f'right {cur_angle-180}'
        # last_angle_msg = f'left {self.last_angle-180}' if self.last_angle > 180 else f'right {self.last_angle}'
        # FIXME: 这里没有搞init observation这种 ; viewpointID乱给的; 没有搞angle的变化信息
        HISTORY_PROMPT = f'History Observation:{viewpoint_summary}\nThought: {thought}Action: action_maker\nAction Input: \"{last_action}\"\n'                                                                                   
                            # Action: {action_vp} \n'
        # self.last_angle = cur_angle
        self.viewpoint_summary_cache.append(HISTORY_PROMPT)
    
    def function_runtime(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            print(f"Function {func.__name__} toke {time.time()-start_time} seconds")
            return result
        return wrapper
    
    @function_runtime
    def parse_observation_prompt(self, observation, batch_index, batch_candidate_viewpoints):
        angle = 30
        
        cur_angle2degree = observation['cur_angle'].item() / np.pi * 180
        viewpointId_offset = round(cur_angle2degree / observation['split_angle'].item())
        def round_angle(x):
            if x > 360:
                x -= 360
            elif x < 0:
                x += 360
            return x
        
        rgb_list = []
        depth_list = []
        num = int(360 / observation['split_angle'].item())
        for i in range(num):
            if i == 0:
                # rgb_list.append(Image.fromarray(observation['rgb'][batch_index].cpu().numpy()))
                rgb_list.append(observation['rgb'][batch_index].permute(2,1,0).cpu())
                depth_list.append(observation['depth'][batch_index].resize_(observation['rgb'][batch_index].size()[0],observation['rgb'][batch_index].size()[1], 1))
            else:
                # rgb_list.append(Image.fromarray(observation[f'rgb_{angle}.0'][batch_index].cpu().numpy()))
                rgb_list.append(observation[f'rgb_{angle}.0'][batch_index].permute(2,1,0).cpu())
                depth_list.append(observation['depth'][batch_index].resize_(observation[f'depth_{angle}.0'][batch_index].size()[0],observation['rgb'][batch_index].size()[1], 1))
                angle += int(observation['split_angle'].item())
                
        # captions = []
        # bboxs = []
        # labels = []
        # batch forward
        rgb_list = torch.stack(rgb_list, dim=0)
        captions, bboxs, labels = self.VFM(rgb_list)
        
        # for rgb in rgb_list:
        #     caption, bbox, label = self.VFM(rgb)
        #     captions.append(caption[0])
        #     bboxs.append(bbox)
        #     labels.append(label)
        
        object_list = []
        object_msg = lambda depth, angle_msg, label: f'[{label}:{angle_msg},{depth:.2f}m]'
        angle_msg = lambda angle: f'left {angle:.2f}' if angle < 180 else f'right {360-angle:.2f}'
        for li in range(len(labels)):
            object = []
            label = labels[li]
            bbox = bboxs[li]
            # lable in [x0, y0, x1, y1]
            for b,l in zip(bbox, label):
                center_xy = [int((b[0]+b[2])/2), int((b[1]+b[3])/2)]
                center_depth = depth_list[li][center_xy[1], center_xy[0]].item() * 10
                angle_offset = (center_xy[0] - int(depth_list[li].size()[0]/2)) / depth_list[li].size()[0] * observation['split_angle'].item()
                angle = li * observation['split_angle'].item() + angle_offset
                angle_msg_obj = angle_msg(round_angle(angle - cur_angle2degree))
                object.append(object_msg(center_depth, angle_msg_obj, l))
            object_list.append(object)
        
        candidate_viewpoints = batch_candidate_viewpoints[batch_index]
        observation_prompt = 'Current Observation:\n\n'
        split_angle = int(observation['split_angle'].item())
        for i in range(num):
            if i in candidate_viewpoints.keys():
                candidate_viewpoints_prompt = 'Navigable viewpoints:\n'
                for cvp in candidate_viewpoints[i]:
                    uid = cvp['unique_id']
                    amsg = angle_msg(int(round_angle((cvp['angle'] / np.pi * 180) - cur_angle2degree)))
                    dis = cvp['distance']
                    candidate_viewpoints_prompt += f'[{uid} : {amsg}, {dis}m]'
            else:
                candidate_viewpoints_prompt = 'Navigable viewpoints: None'
            candidate_viewpoints_prompt += '\n'
            heading2robot = round_angle(split_angle*i - cur_angle2degree)
            observation_prompt += f'Heading : {angle_msg(heading2robot)}\n\
{captions[i]}\n\
Objects in View:{object_list[i]}\n\
{candidate_viewpoints_prompt}\n'

        return observation_prompt
    
    def reset(self, episode_id):
        self.history_chats = []
        self.viewpoint_summary_cache = []
        self.episode_id = episode_id
        if hasattr(self, "logger"):
            del self.logger
        self.logger = logging.getLogger(__name__)

        # handler1 = logging.StreamHandler()
        # handler1.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        # handler1.setFormatter(formatter)
        # self.logger.addHandler(handler1)
        self.logger.setLevel(logging.DEBUG)
        if not os.path.exists(f'{BASE_LOG_DIR}/Prompt'):
            os.makedirs(f'{BASE_LOG_DIR}/Prompt')
        file_handler = logging.FileHandler(f'{BASE_LOG_DIR}/Prompt/{episode_id}.log')
        file_handler.setLevel(level=logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.info(f'###################################################{episode_id}#########################################\n')
        
    def NavGPT_prompt(self, 
                        instruction, 
                        batch_index, 
                        observation, 
                        batch_candidate_viewpoints,
                        thought=['Thought: I should start navigation according to the instruction,',
                                 'Thought:']
                    ):
        
        sys_prompt = self.parse_system_prompt(instruction)
        
        if not len(self.viewpoint_summary_cache) == 0:
            history_prompt = ''
            for hp in self.viewpoint_summary_cache[-10:]:
                history_prompt += f'{hp}\n'
        else:
            history_prompt = '\n'
        history_prompt = 'History:\n' + history_prompt
        
        observation_prompt = self.parse_observation_prompt(observation, batch_index, batch_candidate_viewpoints)
        
        thought_prompt = thought[0] if len(self.viewpoint_summary_cache) == 0 else thought[1] 
        
        format_prompt = 'Please strictly follow the output format:\nThought: you should always think about what to do next and why\n\
Action: the action to take, must be one of the tools [action_maker]\nAction Input: "Viewpoint ID in current observation"\n'
        
        overall_prompt = sys_prompt + history_prompt + observation_prompt + format_prompt + thought_prompt
        
        # if len(overall_prompt) >= 3500:
        #     history_prompt = ''
        #     for hp in self.viewpoint_summary_cache[:-10]:
        #         history_prompt += f'{hp}\n'
        #     overall_prompt = sys_prompt + history_prompt + observation_prompt + format_prompt + thought_prompt
        
        return overall_prompt, observation_prompt
    
    def _answer2info(self, answer, step=None):
        try:
            if 'Finished' in answer:
                action = 'finish'
                through = 'none'
            else:
                action_tool_index = answer.find('Action:')
                through = answer[:action_tool_index]    
                action_start_index = answer.find('Action Input')
                action_str = answer[action_start_index:]
                action = action_str[len('Action Input: '):].split('"')[1]
        except Exception as e:
            print("Parse Action And Though Error:", e)
            action = 'fail'
            through = 'none'
        
        self.logger.info(f'------------------------------------action---------------------------------------------')
        self.logger.info(action)    
        return action, through
    
    def forward(self, overall_prompt, observation_prompt):
        self.logger.info(f'---------------------------------------step{self.step}------------------------------------------------')
        self.step += 1
        overall_message = self._prompt2message(overall_prompt)
        self.logger.info('---------------------------------------query------------------------------------------------')
        self.logger.info(overall_prompt)
        answer, token_info = self.query_gpt3(overall_message)
        self.logger.info('---------------------------------------answer------------------------------------------------')
        self.logger.info(answer)
        action, thought = self._answer2info(answer)
        # if (not action == 'fail'):
        #     self.parse_history_message(observation_prompt, thought)
        return action, thought, token_info
    
    
    
if __name__ == '__main__':
    t = NavGPT()
    print('start query')
    msg = t._prompt2message('hello')
    r,_ = t.query_gpt3(msg)
    print(r)

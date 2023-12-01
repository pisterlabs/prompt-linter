import ast
import guidance
from dotenv import load_dotenv, find_dotenv
import os
import openai
from pydantic import BaseModel, Field, validator
from typing import Optional
import sys
sys.path.append("/Users/allyne/Documents/GitHub/Unity-Agent/")
import agent as A

'''
class Output(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    task: str
    feedback: Optional[str] = None
    plan: list[str]
    functions: list[str]
    script: str
'''

class Critic:
    # gpt-3.5-turbo-1106
    # gpt-4-0613
    def __init__(self, model_name="gpt-4-0613", temperature=0, resume=False, ckpt_dir="ckpt"):
        load_dotenv(find_dotenv())
        openai.api_key = os.getenv("OPENAI_API_KEY")
        guidance.llm = guidance.llms.OpenAI(model_name, temperature=temperature)
        self.llm=guidance.llm
    
    def _refine_plan(self, output_object, examples):
        # Generate new plan, only changing where necessary
        refiner = guidance('''
        {{#system~}}
        You are an efficient, direct and helpful assistant tasked with helping shape a ubicomp space in Unity. Your current task is to refine an existing plan based on specific user feedback.
        {{~/system}}
        {{#user~}}
        Based on the user's prompt, "{{output_object.task}}", you generated a set of instructions, "\n{{output_object.plan}}". 
        The user has provided feedback for you to refine the instructions: "{{output_object.feedback}}".
        
        Your role now is to revise the original set of instructions according to the feedback provided. Follow these guidelines:
        - Review each instruction in the original plan. For those that are still relevant, keep it unchanged and use the exact wording as in the original plan.
        - Identify and modify instructions that are directly affected by the user's feedback. Focus on clarity and specificity in these revisions. You can remove instructions if necessary.
        - Maintain the atomic nature of each instruction, with each directive modifying only 1 property or behavior.               
        - Properties that can be edited are: Position, Rotation, Size (x, y, z), Color, Illumination (Whether the object emanates light), Luminous Intensity (Brightness of the light between 1 and 10), Levitation (When an object is not levitated, it follows the rules of gravity, and when levitated, it floats). 
        - For colors, use RGBA values.
        - Important: Your instructions must translate all subjective or vague terms from the feedback into quantifiable measures. 

        Some examples of translating subjective terms into quantifiable measures. Do not copy these examples, but use them as a guide for your own revisions:
            - If the feedback is to make an object "shorter", modify the size instruction to specify exact measurements, such as explicitly stating to adjust the X and Z axis with the original stretch factor but modify the Y axis' stretch factor (e.g., reduce to only 0.5 times its current size).
            - If the request is to move an object "closer to me", adjust the position instruction to be an added 0.2m closer to the user" or an appropriate relative distance.
        
        Here are examples of similar instructions which may or may not be applicable to you.
        \n {{examples}}
                           
        The format for the revised plan should be:
        1. [Revised or original instruction 1]\n
        2. [Revised or original instruction 2]\n
        …

        *Note: The revised output should integrate the user's feedback while preserving as much of the original plan as possible.*
        {{/user}}
        {{#assistant~}}
        {{gen "new_steps" max_tokens=1000 temperature=0}}
        {{/assistant}}
        ''')
        resp = refiner(output_object=output_object, examples=examples)
        return resp["new_steps"]


    def _old_refine_plan(self, output_object):
        # Identify which part of the plan should be changed (list of int representing index)
        # Identify the change of the plan  
        # Generate a new plan for that part 
        # Use generate function given new step, replace the index in the output object's functions
        # Call create script to create new script
        refiner = guidance('''
        {{#system~}}
        You are an efficient, direct and helpful assistant tasked with refining a plan for shaping a ubicomp space you provided earlier, based on user feedback. 
        {{~/system}}
        {{#user~}}
        Based on the user's prompt, {{output_object.task}}, you generated a plan, {{output_object.plan}}.
        The user has provided a feedback for you to refine the plan: {{output_object.feedback}}.
        Identify the steps in the plan that should be changed based on this feedback. 
        Return a list of indices (0-based) corresponding to the steps that need revision. 
        For example, if steps 1 and 2 need changes, return [0, 1].
        You MUST only return a list of integers.
        {{~/user}}
        {{#assistant~}}
        {{gen "index_list" max_tokens=20 temperature=0}}
        {{~/assistant}} 
        {{#user~}}
        For the steps that need revision, refine the steps to better meet the user's expectations.
        Here are some notes you should take note of as you are regenerating the steps:
        a. Stick to coding conventions, avoiding GUI terms like 'drag' or 'click'.
        b. Be precise in your instruction. Derive numbers from the room and objects unless specified the user.
        c. Translate vague size terms like "big" or "small" into specific multipliers or measurements. For instance, "big" could translate to "2 times its current size", and "small" could translate to "0.5 times its current size", based on the properties of the scene and objects.
        d. Use specific math expressions for vague terms, e.g., instead of "close to the desk", use "smaller than <math expression based on room and object size>".
        e. Adjust the orientation of objects placed on non-horizontal surfaces such as walls to fit that surface.
        f. Every object can be illuminated, so you can use any of them as lights.
        g. Your instruction should not be in a code format. These instruction should be easy to understand.
        h. The instructions must be numbered and in each number only one action. 
        h. You must not respond anything other than the JSON. Do not add any text to before or after the JSON.
        i. If at any point the user mentions an object, and there are no objects remotely close to what they said in the list of current objects in the room given to you, you should make the instruction null and explain in the message.
        j. When referring to changes in properties, always provide specific multipliers or measurements based on the properties of the scene and objects. Avoid vague terms without accompanying tangible values.
                           
        Please give the new refined step for each of the index in the list you provided earlier.
        ---{{#each index_list}}
        Option {{@index}}: {{output_object.plan[this]}}{{/each}}
        Your output should strictly follow the format below. Do NOT include any other information in your output.
        For example, if you have identified index 0 and 2 earlier, you can return the following:
        1. Step 2\n
        3. Step 4\n            
        ...
        {{~/user}}
        {{#assistant~}}
        {{gen "new_steps" max_tokens=1000 temperature=0}}
        {{/assistant}}
        ''')
        resp = refiner(output_object=output_object)
        return resp["index_list"], resp["new_steps"]
        # return output_object
    
    def _refine_code(self, output_object):
        # From debug statements, identify which functions need to be changed
        return output_object
    
    def string_to_list(input_string):
        try:
            result = ast.literal_eval(input_string)
            if isinstance(result, list) and all(isinstance(i, int) for i in result):
                return result
            else:
                raise ValueError("Input string does not represent a list of integers.")
        except (SyntaxError, ValueError):
            raise ValueError("Invalid input string.")


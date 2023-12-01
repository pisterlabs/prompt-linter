from dotenv import load_dotenv, find_dotenv
import guidance
import openai
import os

class Planner:
    # gpt-3.5-turbo-1106
    # gpt-4-0613
    def __init__(self, model_name="gpt-4-0613", temperature=0.0, resume=False, ckpt_dir="ckpt"):
        load_dotenv(find_dotenv())
        openai.api_key = os.getenv("OPENAI_API_KEY")
        guidance.llm = guidance.llms.OpenAI(model_name, temperature=temperature)
        self.llm=guidance.llm

    def _generate_plan(self, task, examples):
        planner = guidance('''
        {{#system~}}
        You are an efficient, direct and helpful assistant tasked with helping shape a ubicomp space. 
        Your role is to generate clear, precise, and effective instructions for altering a 3D space according to user requests.
        When responding to user requests, you must closely follow past successful examples provided. 
        Your instructions should replicate the steps from these examples as closely as possible, only deviating slightly if necessary to tailor the plan to the specific user request. 
        {{~/system}}
        {{#user~}}
        As an assistant, create clear and precise instructions to alter a 3D ubicomp space according to user requests. 
                           
        Follow these guidelines:
        - Respond with a numbered set of instructions. 
        - Your first instruction must be to either create an object or find the object the user is referring to.
        -- For example, if the user uses phrases like "The table" and "This table", you should have an instruction like "Find a table in the user's field of view"
        - Each instruction should modify only 1 property or behaviour.
        - Properties that can be edited are: Position, Rotation, Size, Color, Illumination (Whether the object emanates light), Luminous Intensity (Brightness of the light between 1 and 10), Levitation (When an object is levitated, it floats). 
        - If you need to edit the position of more than one object, include it within a single instruction. For example, use "Edit the Position property of each chair to be 0.5 meters in front of each room wall" instead of separate instructions for each chair.
        - Your instructions must translate subjective terms into specific, measurable instructions. 
        -- For example, terms like "big" or "close to me" can translate to “2 times its current size” and  “1m away from the user” respectively. Always cite explicit numbers.
        -- Terms like "the table" or "this table" should translate to "table in the user's field of view"
        - For colors, use RGBA values.
        - Only instructions modifying the Position property can mention more than one object types. All other property modification can only mention ONE object type.
        
        The space consists of 4 walls, 1 ceiling, and 1 floor.
        
        You are limited to creating or modifying the following object types: (You must use the exact case-sensitive type of the object)
        Chair, Fox, Lamp, LED Cube, Push Button, Table, Vase, Zombie

        The user's prompt is {{task}}.
                                
        When presented with a user request, your first step is to compare it with past examples provided below.
        If the current request closely matches a past example, you must replicate the plan from that example as closely as possible, adjusting only what is necessary to fit the specifics of the new task. This replication should include using the same object types, properties, and values.
        For any task, where a past example exists, your plan should follow the example's structure and steps very closely.
        Only in cases where no past example matches closely, then construct a new plan by synthesizing elements from the examples that are most relevant to the new task. 
        Remember, the goal is to maintain the effectiveness and consistency of past successful plans. Use them as blueprints for your responses, ensuring that similar tasks yield similar, proven results.
        
        Past examples:\n {{examples}}

        The format for response should strictly be:
            1. Instruction 1\n
            2. Instruction 2\n
            …

        *Note: Output should not contain any text other than the instructions.*
        {{~/user}}
        {{#assistant~}}
        {{gen "plan" max_tokens=2000 temperature=0}}
        {{~/assistant}}
        ''')
        resp = planner(task=task, examples=examples)
        return resp["plan"]
    
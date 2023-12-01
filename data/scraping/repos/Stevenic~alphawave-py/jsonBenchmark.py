import argparse
import json
import time
import aiounittest, unittest
import os, sys
from promptrix.promptrixTypes import Message, PromptFunctions, PromptMemory, Tokenizer
from promptrix.SystemMessage import SystemMessage
from promptrix.UserMessage import UserMessage
from promptrix.ConversationHistory import ConversationHistory
from promptrix.AssistantMessage import AssistantMessage
from promptrix.FunctionRegistry import FunctionRegistry 
from promptrix.Prompt import Prompt
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.VolatileMemory import VolatileMemory
from alphawave.alphawaveTypes import PromptCompletionOptions, PromptResponse, PromptResponseValidator, Validation
from alphawave.DefaultResponseValidator import DefaultResponseValidator
from alphawave.JSONResponseValidator import JSONResponseValidator
from alphawave.RepairTestClient import TestClient as TestClient
from alphawave.AlphaWave import AlphaWave
from alphawave.OpenAIClient import OpenAIClient
import alphawave_pyexts.LLMClient as llm
from alphawave_pyexts.SearchCommand import SearchCommand
from alphawave.OSClient import OSClient
from alphawave_agents.Agent import Agent, AgentOptions
from alphawave_agents.AskCommand import AskCommand
from alphawave_agents.MathCommand import MathCommand
from alphawave_agents.FinalAnswerCommand import FinalAnswerCommand

# Create an OpenAI client
#client = OpenAIClient(apiKey=os.getenv('OPENAI_API_KEY'))#, logRequests=True)
model = 'gpt-3.5-turbo'
#model = 'chatglm2'

#create OS client
client = OSClient(apiKey=os.getenv('OPENAI_API_KEY'))#, logRequests=True)

chat_prime = 'You are a friendly, helpful, courteous AI named John.'

json_prime = """
Reason step by step how to answer the users query below.
use the following JSON format with your thoughts and the answer:
Response Format:
{"reasoning": '<reflections on how to construct an answerto the user query>',"answer": '<answer to user query'}}

"""

pre_agent_prime = """
Reason step by step about the input.
You have the following actions available:
 - ask - ask the user a question
 - math - evaluate a python mathematical expression
 - answer - present an answer to the user. The answer must respond to the original user input
chose the action to use next.
Finally, build the input to pass to the action.
Return a JSON object with your reasoning, action, and input, using the following format:
Response Format:

{"reasoning": '<concise plan for developing answer>',"action": '<name of action to use next>', "inputs": '<action inputs>'}}
"""


planning_prime="""
Reason step by step about the user task.
You have the following actions available:
 - ask - ask the user a question
 - math - evaluate a python mathematical expression
 - answer - present an answer to the user. The answer input must respond to the original user input
Develop a concise plan to perform the task and present the result to the user.
Return a JSON object with your reasoning and plan. Use the following format.
Response Format:

{"reasoning": '<thoughts about the task>', "plan":'<actions to accomplish task>'}
"""


chat_prompt =Prompt([SystemMessage(chat_prime),
                     ConversationHistory('history'),
                     UserMessage("{{$input}}")])  #,AssistantMessage('')])

zeroShot_prompt=Prompt([SystemMessage(json_prime),
                        ConversationHistory('history'),
                        UserMessage("{{$input}}")]) #,AssistantMessage('')])

oneShot_prompt=Prompt([SystemMessage(json_prime),
                       UserMessage("What is 2 + 2"),
                       AssistantMessage('{"reasoning":"That is easy, I know the answer", "answer":"4"}'),
                       ConversationHistory('history'),
                       UserMessage("{{$input}}")])#, AssistantMessage('')])

planning_prompt=Prompt([SystemMessage(planning_prime),
                     UserMessage("What is 2 + 2"),
                     AssistantMessage('{"reasoning":"I am not great at math, and I need to answer the user.", "plan":"math"}'),
                     ConversationHistory('history'),
                     UserMessage("{{$input}}")])#, AssistantMessage('')])

pre_agent_prompt=Prompt([SystemMessage(pre_agent_prime),
                     UserMessage("What is 2 + 2"),
                     AssistantMessage('{"reasoning":\"I am not great at math, better use the math action\", "action":"math", "input":"2 + 2"}'),
                     ConversationHistory('history'),
                     UserMessage("{{$input}}")])#, AssistantMessage('')])

agent_prompt=Prompt([
                     UserMessage("What is 2 + 2"),
                     AssistantMessage('{"reasoning":"I am not confident of my math ability", "plan":"use the math command", "command":{"name":"math", "inputs":{"code":"2 + 2"}}}')
])


validation_schema={
    "type": "object",
    "properties": {
        "reasoning": {
            "type":"string",
            "description": "<reasoning about problem>"
        },
        "answer": {
            "type":"string",
            "description": "<response to query>"
        }
    },
    "required": ["reasoning", "answer"]
}

planning_validation_schema={
    "type": "object",
    "properties": {
        "reasoning": {
            "type":"string",
            "description": "<reasoning about problem>"
        },
        "plan": {
            "type":"string",
            "description": "<actions in plan>"
        }
    },
    "required": ["reasoning", "plan"]
}

pre_agent_validation_schema={
    "type": "object",
    "properties": {
        "reasoning": {
            "type":"string",
            "description": "<reasoning about problem>"
        },
        "action": {
            "type":"string",
            "description": "<name of chosen action>"
        },
        "input": {
            type: "string",
            "description": "<input for chosen action>"
        },
    },
    "required": ["reasoning", "action", "input"]
}

agent_validation_schema={
    "type": "object",
    "properties": {
        "reasoning": {
            "type":"string",
            "description": "<reasoning about problem>"
        },
        "action": {
            "type":"string",
            "description": "<name of chosen action>"
        },
        "input": {
            type: "string",
            "description": "<input for chosen action>"
        },
    },
    "required": ["reasoning", "action", "input"]
}

def make_agent(model):
    agent_options = AgentOptions(
        client = client,
        prompt=["You are a helpful information bot, eager to answer questions.\n" ],
        prompt_options=PromptCompletionOptions(model = model, temperature = 0.2, max_input_tokens = 1200, max_tokens = 400 ),
        syntax='JSON'
    )
    agent = Agent(options = agent_options)
    return agent
    


class TestAlphaWave():
    def __init__(self, name):
        #model = input('template_name? ').strip()
        self.name = name
        self.model=model
        self.client = client
        self.prompt = oneShot_prompt
        self.prompt_options = PromptCompletionOptions(completion_type='chat', model=self.model, temperature=0.1, max_tokens=100)
        self.memory = VolatileMemory()
        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()
        self.validator = JSONResponseValidator(validation_schema)
        self.ok = True
        self.record = []

    def assertTrue(self, x):
        if not x:
            #print(f'### {self.name} failed')
            self.ok = False
            
    def end(self, msg = ''):
        if self.ok:
            print(f'### {self.name} passed {msg}')
        else:
            print(f'### {self.name} failed {msg}')
        self.ok=True # reset
            
    def test_chat(self):
        print('\n************************************************************************************************')
        wave = AlphaWave(client=self.client, prompt=chat_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=DefaultResponseValidator())
        self.memory.clear()
        self.memory.set('history', [])
        self.memory.set('input', 'Hello. How are you today?')
        response = wave.completePrompt()
        print(response)
        self.assertTrue(response['status'] =='success')
        self.end()

    def test_chat_recall(self):
        print('\n************************************************************************************************')
        wave = AlphaWave(client=self.client, prompt=chat_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=DefaultResponseValidator())
        self.memory.set('history', [])
        self.memory.set('input', 'Hello. How are you today?')
        response = wave.completePrompt()
        self.assertTrue(response['status'] == 'success')
        self.assertTrue(len(self.memory.get('history')) == 2)
        self.memory.set('input', 'What is your name?')
        #print(f"history: {self.memory.get('history')}")
        response = wave.completePrompt()
        self.assertTrue(len(self.memory.get('history')) == 4)
        self.assertTrue(response['status'] == 'success')
        print(response['message']['content'])
        self.assertTrue('John' in response['message']['content'])
        self.end()

    def test_reasoning(self):
        print('\n************************************************************************************************')
        wave = AlphaWave(client=self.client, prompt=chat_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=DefaultResponseValidator())
        self.memory.clear()
        self.memory.set('history', [])
        self.memory.set('input', """Situation: In the room there are John, Mark, a cat, a box, and a basket. John takes the cat and puts it in the basket. He leaves the room and goes to school. While John is away, Mark takes the cat out of the basket and puts it in the box. Mark leaves the room and goes to work. John comes back from school and enters the room. He doesnt know what happened in the room when he was away.

Question: Where does John think the cat is when he re-enters the room??""")
        response = wave.completePrompt()
        self.assertTrue(response['status'] == 'success')
        self.assertTrue('basket' in response['message']['content'])
        print(response['message']['content'])
        self.end()

    def test_json_zeroShot(self):
        print('\n************************************************************************************************')
        wave = AlphaWave(client=self.client, prompt=zeroShot_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)
        self.memory.clear()
        self.memory.set('history', [])
        self.memory.set('input', 'Hello. How are you today?')
        response = wave.completePrompt()
        self.assertTrue(response['status'] == 'success')
        print(response)
        self.end()

    def test_json_zeroShot_2turn(self):
        print('\n************************************************************************************************')
        wave = AlphaWave(client=self.client, prompt=zeroShot_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)
        self.memory.clear()
        self.memory.set('history', [])
        self.memory.set('input', 'Hello. How are you today?')
        response = wave.completePrompt()
        self.assertTrue(response['status'] == 'success')
        self.memory.set('input', 'Hello. What is your name?')
        response = wave.completePrompt()
        self.assertTrue(response['status'] == 'success')
        print(response)
        #assert_that(response['message']).is_equal_to({ 'role': 'assistant', 'content': 'Hello' })
        self.end()

    def test_json_oneShot(self):
        print('\n************************************************************************************************')
        self.memory.clear()
        self.memory.set('history', [])
        wave = AlphaWave(client=self.client, prompt=oneShot_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)
        self.validator.repairAttempts = 0
        response = wave.completePrompt('Hello. How are you today?')
        self.assertTrue(response['status'] == 'success')
        print(response)
        self.end()
        
    def test_json_oneShot_repair(self):
        print('\n************************************************************************************************')
        self.memory.clear()
        self.memory.set('history', [])
        wave = AlphaWave(client=self.client, prompt=oneShot_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=self.validator)

        self.validator.repairAttempts = 2
        response = wave.completePrompt('Hello. How are you today?')
        print(response)
        self.assertTrue(response is not None)
        if response is not None:
            self.assertTrue('status' in response)
            if 'status' in response:
                self.assertTrue(response['status'] == 'success')
        self.end()
        
    def test_planning(self):
        print('\n************************************************************************************************')
        # Add core actions to the agent
        self.memory.clear()
        self.memory.set('history', [])
        wave = AlphaWave(client=self.client, prompt=planning_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=JSONResponseValidator(planning_validation_schema))
        self.validator.repairAttempts = 1
        response = wave.completePrompt('What is the square root of 2?')
        self.assertTrue(response['status'] == 'success')
        print(response)
        self.end('format')
        self.assertTrue('message' in response and 'content' in response['message'] and 'plan' in response['message']['content'] and 'math' in response['message']['content']['plan'])
        self.end('content')

    def test_pre_agent(self):
        print('\n************************************************************************************************')
        # Add core actions to the agent
        self.memory.clear()
        self.memory.set('history', [])
        wave = AlphaWave(client=self.client, prompt=pre_agent_prompt, prompt_options=self.prompt_options, memory=self.memory, functions=self.functions, tokenizer=self.tokenizer, validator=JSONResponseValidator(pre_agent_validation_schema))
        self.validator.repairAttempts = 1
        response = wave.completePrompt('What is the square root of 2?')
        print(response)
        self.assertTrue(response['status'] == 'success')
        self.end('format')
        self.assertTrue('math' in response['message']['content']['action'])
        self.assertTrue('sqrt' in response['message']['content']['input'])
        self.end('content')

    def test_agent_Math(self):
        print('\n************************************************************************************************')
        # Add core actions to the agent
        self.memory.clear()
        self.memory.set('history', [])
        agent = make_agent(self.model)
        agent.addCommand(MathCommand())
        self.memory.clear()
        print('***** benchmark calling agent complete task')
        result = agent.completeTask('You are not good at math, and should use the math command when asked a math question. what is 73 * 21?')
        print(result)
        if type(result) == dict and 'type' in result and result['type'] == 'TaskResponse':
            self.assertTrue(result['status'] == 'success' or result['status'] == 'input_needed')
            print(result['message'])
            self.assertTrue('1533' in str(result['message']))
        else:
            print(f' Unknown Agent result type: {result}')
            self.assertTrue(False)
        self.end()

    def test_agent_Search(self):
        print('\n************************************************************************************************')
        # Add core actions to the agent
        self.memory.clear()
        self.memory.set('history', [])
        agent = make_agent(self.model)
        agent.addCommand(SearchCommand(OpenAIClient(apiKey=os.getenv('OPENAI_API_KEY')), model='gpt-3.5-turbo'))
        #agent.addCommand(FinalAnswerCommand())
        self.memory.clear()
        print('***** benchmark calling agent complete task')
        result = agent.completeTask('weather forecast for Berkeley, Ca?')
        if type(result) == dict and 'type' in result and result['type'] == 'TaskResponse':
            self.assertTrue(result['status'] == 'success' or result['status'] == 'input_needed')
            print(result['message'])
            self.end('format')
            self.assertTrue('temp' in str(result['message']))
            self.end('content')
        else:
            print(f' Unknown Agent result type: {result}')
            self.assertTrue(False)
            self.end('format')

if __name__ == '__main__':
    if len(sys.argv)<2:
        modelin = input('model name? ').strip()
        if modelin is not None and len(modelin)>1:
            model = modelin
            models = llm.get_available_models()
            while model not in models:
                print(models)
                modelin = input('model name? ').strip()
                model=modelin
    else:
        model = sys.argv[1].strip()
        if model not in llm.get_available_models():
            print(f' Fail, model template {model} not known')
            sys.exit(-1)
    start_time = time.time()
    test = TestAlphaWave('chat')
    test.test_chat()
    test = TestAlphaWave('recall')
    test.test_chat_recall()
    test = TestAlphaWave('reasoning')
    test.test_reasoning()
    # last because some models drift off into hyperspace
    #test = TestAlphaWave('json_zeroShot')
    #test.test_json_zeroShot()
    #test = TestAlphaWave('json_zeroShot_2turn')
    #test.test_json_zeroShot_2turn()
    #test = TestAlphaWave('json_oneShot no repair')
    #test.test_json_oneShot()
    test = TestAlphaWave('json')
    test.test_json_oneShot_repair()
    test = TestAlphaWave('action_selection')
    test.test_planning()
    test = TestAlphaWave('pre_agent')
    test.test_pre_agent()
    test = TestAlphaWave('agent')
    test.test_agent_Math()
    #test = TestAlphaWave('test_agent Search')
    #test.test_agent_Search()
    elapsed_time = int(time.time()-start_time)
    print(f'### Time {elapsed_time}')
    time.sleep(1)

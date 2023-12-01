from torch import cuda, bfloat16
import transformers
import sys
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate

from langchain import OpenAI, LLMChain, LLMMathChain
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
import langchain
from transformers import GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.tools import ShellTool
from langchain.tools import HumanInputRun
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
from datasets import load_dataset
from langchain.llms import HuggingFacePipeline

import chess
import chess.engine
from stockfish import Stockfish
import re

from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

sys.path.append('/home/jovyan/.local/lib/python3.11/site-packages')
sys.path.append('/home/jovyan/.local/bin')

model_id = 'meta-llama/Llama-2-70b-chat-hf'
    
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, need auth token for these
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)
print("loading model")
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()
print(f"Model loaded on {device}")
print("loading tokenizer")
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)
print("loaded tokenizer")
generate_text = transformers.pipeline(
    model=model, 
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

llm = HuggingFacePipeline(pipeline=generate_text)
local_llm = llm

search = DuckDuckGoSearchRun()
#wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
shell_tool = ShellTool()
shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace(
    "{", "{{"
).replace("}", "}}")
self_ask_with_search = initialize_agent(
    [shell_tool], local_llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=False
)

def duck_wrapper(input_text):
#   search_results = search.run(f"site:webmd.com {input_text}")
    search_results = search.run(f"site:google.com {input_text}")
    return search_results

def chess_guide(input_text):
    search_results2 = search.run(f"site:chess.com {input_text}")
    return search_results2

def shell(input_text):
    search_results4 = self_ask_with_search.run(input_text)
    return search_results4

def chess_moves(input_text):
#    stockfish = Stockfish('/workspace/pv-data/InternFolders/Niko/stockfish-ubuntu-x86-64-modern')
    stockfish = Stockfish('/workspace/pv-data/stockfish-ubuntu-x86-64-modern')
    input_text = input_text.replace("\"", "")
    moves = input_text.split()
    moves_clean = []
    for i in moves:
        x = i.find(".")
        if x != -1:
            moves_clean.append(i[x+1:])
        else:
            moves_clean.append(i)

    board = chess.Board()
    for move in moves_clean:
        board.push_san(move)
    fen = board.fen()

    stockfish.set_fen_position(fen)

    best_move = stockfish.get_best_move()

    return (f"The next best move according to Stockfish is: '{best_move}'")

class HMT():
    def __init__(self):
        # Set up the base template
        template = """Answer the following questions as best you can, but speaking as a professional chess player. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Here are some examples for this format:

        Question: How old was Albert Einstein when he died divided by two
        Thought: I should find out how old Albert Einstein was when he died. This is a general question, so I should use the Search DukcDuckGo tool
        Action: Search DuckDuckGo
        Action Input: "How old was Albert Einstein when he died."
        Observation: (Tool Result) Albert Einstein was 76 years old when he died
        Thought: I now know the final answer
        Final Answer: Albert Einstein's age when he died is 76

        Question: I am playing a chess game with the following moves in algebraic notation: e4 e5 f4. I am black, what should be my next move?
        Thought: This is a specific chess question related to an ongoing game. I should use the Chess Move Predict tool
        Action: Chess Move Predict
        Action Input: "e4 e5 f4"
        Observation: (Tool Result) Black should play e5 to f4
        Thought: I now know the final answer
        Final Answer: My next move should be e5 to f4

        Question: What is 8 multiplied by 64?
        Thought: This is an arithmetic problem, I should use the Calculator tool.
        Action: Calculator
        Action Input: "What is 8 multiplied by 64?"
        Observation: (Tool Result) 8 multiplied by 64 is 512
        Thought: I now know the final answer
        Final Answer: 8 multiplied by 64 is 512

        Question: What happens to the pawn when it reaches the end of the chess board?
        Thought: This is a general question related to chess, I should use the Chess Search tool
        Action: Chess Search
        Action Input: "What happens to the pawn when it reaches the end of the chess board?"
        Observation: (Tool Result) Pawn Promotion is one of the special moves in chess. It happens when a Pawn reaches the opponent's back rank (first row of the opponent) and then it is replaced by any piece a player decides to, except The King
        Thought: I now know the final answer
        Final Answer: Pawn Promotion is one of the special moves in chess. It happens when a Pawn reaches the opponent's back rank (first row of the opponent) and then it is replaced by any piece a player decides to, except The King

        Begin! Remember to answer as a professional chess player and use the template when giving your final answer.

        Question: {input}
        {agent_scratchpad}"""
        
        llm_math_chain = LLMMathChain.from_llm(llm=local_llm, verbose=False)
        tools = [
            Tool(
                name = "Google Search",
                func=duck_wrapper,
                description="useful for getting answers to general questions"
            ),
            Tool(
                name="Calculator",
                func=llm_math_chain.run,
                description="useful for when you need to answer arithmetic questions"
            ),
            Tool(
                name="Chess Search",
                func=chess_guide,
                description="useful for general chess questions related to openings or pieces"
            ),
            Tool(
                name="Shell Tool",
                func=shell,
                description = "useful for interacting with the local file system and using shell commands"
            ),
            Tool(
                name="Chess Move Predict",
                func=chess_moves,
                description="useful for predicting the next chess move when given a series of moves in standard algebraic notation"
            )
        ]
        
        # Set up a prompt template
        
        class CustomPromptTemplate(StringPromptTemplate):
            # The template to use
            template: str
            # The list of tools available
            tools: List[Tool]

            def format(self, **kwargs) -> str:
                # Get the intermediate steps (AgentAction, Observation tuples)
                # Format them in a particular way
                intermediate_steps = kwargs.pop("intermediate_steps")
                thoughts = ""
                for action, observation in intermediate_steps:
                    thoughts += action.log
                    thoughts += f"\nObservation: {observation}\nThought: "
                # Set the agent_scratchpad variable to that value
                kwargs["agent_scratchpad"] = thoughts
                # Create a tools variable from the list of tools provided
                kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
                # Create a list of tool names for the tools provided
                kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
                return self.template.format(**kwargs)
            
        prompt = CustomPromptTemplate(
            template=template,
            tools=tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps"]
        )
        
        class CustomOutputParser(AgentOutputParser):
    
            def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
                # Check if agent should finish
                if "Final Answer:" in llm_output:
                    return AgentFinish(
                        # Return values is generally always a dictionary with a single `output` key
                        # It is not recommended to try anything else at the moment :)
                        return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                        log=llm_output,
                    )
                # Parse out the action and action input
                regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
                match = re.search(regex, llm_output, re.DOTALL)
                if not match:
                    raise ValueError(f"Could not parse LLM output: `{llm_output}`")
                action = match.group(1).strip()
                action_input = match.group(2)
                # Return the action and action input
                return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
            
        output_parser = CustomOutputParser()
        # LLM chain consisting of the LLM and a prompt
        llm_chain = LLMChain(llm=local_llm, prompt=prompt)
        tool_names = [tool.name for tool in tools]

        agent = LLMSingleActionAgent(
            llm_chain=llm_chain, 
            output_parser=output_parser,
            stop=["\nObservation:"], 
            allowed_tools=tool_names
        )
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, 
                                                   tools=tools, 
                                                   verbose=False,
#                                                    handle_parsing_errors=True,
                                                   return_intermediate_steps=True)
    def predict(self, query):
        langchain.debug = False
        langchain.verbose = False
        response = self.agent_executor(query)
        ## Parse the string from intermed_result
        keywords = ["log=", "Action:", "Action Input:"]
        result = []
        print(response["intermediate_steps"])
        for input_string_temp in response['intermediate_steps']:
            input_string = str(input_string_temp[0])
            obv = str(input_string_temp[1])
            for index, keyword in enumerate(keywords):
                start_index = input_string.find(keyword)
                if keyword=="log=":
                    end_index = input_string.find(keywords[index+1])
                    appendable = ("Thought:", re.sub(r'[^A-Za-z0-9 ."]+', '', input_string[start_index+len(keyword)-1:end_index-1].strip().replace("Answer:", "").replace("Thought:", "").replace("\\n", "").replace("\\n\\", ""))[1:])
                    result.append(appendable)
                elif keyword == "Action Input:":
                    appendable = (keyword, re.sub(r'[^A-Za-z0-9 ."]+', '', input_string[start_index+len(keyword)-1:len(input_string)].strip().replace("\\n", "").replace("\\n\\", "")))
                    result.append(appendable)
                else:
                    end_index = input_string.find(keywords[index + 1])
                    appendable = (keyword, re.sub(r'[^A-Za-z0-9 ."]+', '', input_string[start_index+len(keyword)-1:end_index-1].strip().replace("\\n", "").replace("\\n\\", "")))
                    result.append(appendable)
            result.append(("Action Output:", obv))

        final_output = ''
        for keyword, value in result:
            final_output += f'{keyword} {value} '
        final_ans = response["output"]
        final_output += f"Final Answer: {final_ans} "
        print(final_output)
        
        return final_output
    
# hmt_model = HMT()
# print(hmt_model.predict("What is the mass of Neptune's biggest moon?"))
# print("\n"*3)
# print(output)
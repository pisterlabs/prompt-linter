import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import LLMSingleActionAgent, AgentExecutor

from client.chains.agents import get_initialized_agent
from client.chains.prompts import get_prompt
from client.chains.tools import get_all_tools
from client.chains.models import get_llm
from client.chains.parsers import get_output_parser
from client.chains.callback_handlers import InputOutputAsyncCallbackHandler

class SingleCommandAssistant:
    def __init__(
            self,
            name="Assistant",
            max_tokens=2048,
            temperature=0.0,
            streaming=False,
            callbacks=[],
            verbose=False,
        ):
        self.name = name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.streaming = streaming
        self.callbacks = callbacks
        self.verbose = verbose
        
        self._initialize_llm()
        self._initialize_prompt()
        self._initialize_output_parser()
        self._initialize_chain()
        self._initialize_tools()
        self._initialize_agent()
            
    def _initialize_llm(self):
        self.llm = get_llm(max_tokens=self.max_tokens, temperature=self.temperature, streaming=self.streaming, callbacks=self.callbacks)
        #from langchain import OpenAI
        #self.llm = OpenAI(temperature=0, callbacks=[InputOutputAsyncCallbackHandler()])

    def _initialize_prompt(self):
        self.prompt = get_prompt()
    
    def _initialize_output_parser(self):
        self.output_parser = get_output_parser(parser="column_csv")
    
    def _initialize_chain(self):
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def _initialize_tools(self):
        self.tools = get_all_tools()

    def _initialize_agent(self):
        tool_names = [tool.name for tool in self.tools]
        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain,
            output_parser=self.output_parser,
            stop=["\nObservation:", "<|im_end|><|endoftext|>", "<|im_end|>", "<|endoftext|>", "</s>"],
            allowed_tools=tool_names
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=self.agent, tools=self.tools, verbose=self.verbose, max_iterations=20)
    
    def __call__(self, query):
        return self.process_output(self.agent_executor.run(input=query))
    
    def prompt_llm(self, query, raw=False):
        """
        Execute the `llm` method with the given `query` and process the output based on the `raw` flag.

        Parameters:
            query (str): The query string to be passed to the `llm` method.
            raw (bool, optional): If True, returns the raw output from `llm`. If False (default), processes the output using `process_output`.

        Returns:
            str: The processed output if it exists and `raw` is False. If `raw` is True, returns the raw output. Returns an empty string if no output is available.
        """
        txt = self.llm(query)
        if txt and not raw:
            return self.process_output(txt)
        elif raw and txt:
            return txt
        else:
            return ""
    
    def process_output(self, output):
        """
        Remove special tokens from the given output and perform any necessary pre-processing steps.

        Args:
            output (str): The output to be processed.

        Returns:
            str: The processed output.
        """
        output = output.replace("<|im_end|>", "")
        output = output.replace("<|endoftext|>", "")
        # Add other pre-processing steps here
        return output
    
    def __str__(self):
        return f"{self.name}: {self.max_tokens} tokens @{self.temperature}°"

if __name__ == "__main__":
    assistant = SingleCommandAssistant(temperature=0.0, max_tokens=200, verbose=True, callbacks=[InputOutputAsyncCallbackHandler()])
    queries = [
        "Hello Assistant! How are you today?", # query goes here
        "Please tell me a joke.",
        "Salut. Je m'appelle John. Comment ça va?",
        "Quelle heure est-il?",
        "Quel temps fait-il aujourd'hui?",        
        ]
    for query in queries:
        print(assistant(query))

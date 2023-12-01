from langchain.llms import OpenAI
import os
import lorem

"""
The agent that codes out certain tasks.
Acts as the the intermediate layer between bt-copilot and the llm
"""

class SimpleCodingAgent:
    """
    The simple coding agent just fowards the prompts to the LLM. No tests, no iterations.
    Other agents can extend this to more complex, iterative coding strategies.

    """
    def __init__(self,API_KEY,test_mode = False):
        """
        Constructs a new instance of the simple_coding_agent.
        Args:
            None
        """

        # set OpenAI API key
        self.API_KEY = API_KEY

        # set test_mode variable. If True --> no llm calls are sent and dummy results are returned.
        self.test_mode = test_mode

    def code(self,prompt,temperature):
        '''
        takes input prompt and returns code-snippet.
        :param prompt: string
        :return: code: string
        '''

        if self.test_mode == False:
            # LLM calls are active

            # initiate llm
            llm = OpenAI(openai_api_key = self.API_KEY,
                         max_tokens=2000,
                         temperature = temperature)

            # run llm call
            code_snippet = llm(prompt=prompt)

        else:
        # LLM calls are deactivated. Only dummy code is returned.
            code_snippet = lorem.paragraph()


        return code_snippet

    def simple_LLMcall(self,prompt,temperature):
        '''
        takes input prompt and returns llm response.
        :param prompt: string
        :return: code: string
        '''

        if self.test_mode == False:
            # LLM calls are active

            # initiate llm
            llm = OpenAI(openai_api_key=self.API_KEY,
                         max_tokens=2000,
                         temperature = temperature)

            # run llm call
            llm_response = llm(prompt=prompt)

        else:
            # LLM calls are deactivated. Only dummy code is returned.
            llm_response = lorem.paragraph()

        return llm_response


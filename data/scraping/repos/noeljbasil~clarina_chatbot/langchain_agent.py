import os
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate

from langchain import OpenAI, LLMChain
from langchain.tools import DuckDuckGoSearchRun 

from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
from langchain.memory import ConversationBufferWindowMemory

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

import langchain

def init():
    """Initialise the agent and streamlit page"""
    # Load the OpenAI API key from the environment variable
    dotenv_path = Path('.env')
    load_dotenv(dotenv_path=dotenv_path)
    
    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    # setup streamlit page
    st.set_page_config(
        page_title='Clarina: Your AI Beauty Guru', 
        page_icon='💄', 
        layout='centered',
    )


    st.markdown(""" <style = 'text-align: center;'> .title { font-size:55px ; font-family: 'Clarins'; color: #c20430;} </style> """, unsafe_allow_html=True)
    st.markdown('<p class="title" align="center">Clarina 💄</p>', unsafe_allow_html=True)
    st.markdown(""" <style = 'text-align: center;> .desc { font-size:20px ;font-family: 'Roboto'; font-weight: 550; font-style: italic ; color: #000000;} </style> """, unsafe_allow_html=True)
    st.markdown('<p class="desc" align="center">Your personal beauty guru</p>', unsafe_allow_html=True)



def init_agent()->AgentExecutor:
    """Initialise the agent and return the agent executor
    Args:
        None
    Returns:
        agent_executor: the agent executor"""

    #define tools for agent: seach and LLM
    search = DuckDuckGoSearchRun()

    def duck_wrapper(input_text):
        search_results = search.run(f"site:clarins.co.uk {input_text}")
        return search_results

    tools = [
        Tool(
            name = "Search Clarins",
            func=duck_wrapper,
            description="useful for when you need to answer questions regarding any products on the Clarins website such as user reviews, benefits, product description, Ingredients and similar products"
        ),
    ]

    # Set up the base template
    clarins_template = """Clarina is a professional beauty coach working for Clarins (clarins.co.uk) whos primary tasks is to help users navigate clarins.co.uk and help them get their questions answered ragrding the varios products on the website.

    Clarina always recommends products from clarins.co.uk which are relevant to the topic at hand and NOT of any of other brands.

    Clarina is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of beauty topics. As a beauty coach, Clarina is able to generate expert opions on beauty related topics based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    Clarina is constantly learning and improving, and her capabilities are constantly evolving. She is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions from the cosmetics and beauty domain. Additionally, Clarina is able to generate her own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of beauty and cosmetics topics.

    Overall, Clarina is a powerful assistant who can help with a wide range of beauty and cosmetic related tasks and provide valuable insights and information on a wide range of beauty and cosmetic topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Clarina is here to assist.

    TOOLS:
    ------

    Assistant has access to the following tools:

    {tools}

    To use a tool, please use the following format:

    ```
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    ```

    When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

    ```AI: [your response here]```

    Keep the following in mind while providing the final answer:
        -Make sure to use only the product data available on clarins.co.uk to provide information relating to clarins products. If you do not know the answer, do not halucinate. 
        -Format your answer as markdown and insert URL to the products in the answer when any product is suggested as part of the final answer. Make sure to sperate out the URLs in different lines.
        -Try to upsell products to the consumer by suggesting products or services available on clarins.co.uk that will compliment their purchase or interest.

    Begin! Remember to answer as a professional beauty coach when giving your final answer.

    Previous conversation history:
    {history}

    New input: {input}
    {agent_scratchpad}
    """

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
        
    prompt_with_history = CustomPromptTemplate(
        template=clarins_template,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps", "history"]
        )

    # The output parser is responsible for parsing the LLM output into AgentAction and AgentFinish. This usually depends heavily on the prompt used.
    # This is where you can change the parsing to do retries, handle whitespace, etc
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
            
            clarina_output_prefix_1 = """AI: """
            if clarina_output_prefix_1 in llm_output.strip():
                return AgentFinish(
                    # Return values is generally always a dictionary with a single `output` key
                    # It is not recommended to try anything else at the moment :)
                    return_values={"output": llm_output.split(clarina_output_prefix_1)[-1].strip()},
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

    # choose LLM model
    llm = OpenAI(temperature=0)

    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)

    #set up agent
    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        stop=["\nObservation:"], 
        allowed_tools=tool_names
    )

    # Agent Executors take an agent and tools and use the agent to decide which tools to call and in what order.
    memory=ConversationBufferWindowMemory(k=3)

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        memory=memory
        )
    
    langchain.debug = True

    return agent_executor
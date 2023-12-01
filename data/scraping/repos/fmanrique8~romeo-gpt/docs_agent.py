from langchain.agents import AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish

# Prompt Template
template = """Answer the following question as best you can, using the text chunks provided:

Text Chunks: {text_chunks}

Question: {input}

Begin! The answer should very detailed: 
{agent_scratchpad}"""


# CustomPromptTemplate
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    input_variables: List[str] = ["input", "intermediate_steps", "text_chunks"]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        return self.template.format(**kwargs)


# Set up the prompt
prompt = CustomPromptTemplate(
    template=template, input_variables=["input", "intermediate_steps", "text_chunks"]
)


# Output Parser
class CustomOutputParser(AgentOutputParser):
    @staticmethod
    def parse(llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        else:
            # Return the entire output as the answer if "Final Answer:" is not found
            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log=llm_output,
            )


output_parser = CustomOutputParser()

# Set up LLM
llm = OpenAI(temperature=0)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Define the agent
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, output_parser=output_parser, stop=["<|END|>"], allowed_tools=[]
)


# Function to run the document agent
def documents_agent(text_chunks: str, task: str) -> str:
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=[], verbose=True
    )
    answer = agent_executor.run(task=task, text_chunks=text_chunks, input=task)
    return answer

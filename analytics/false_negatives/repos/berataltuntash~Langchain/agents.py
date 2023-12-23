import os
os.environ['OPENAI_API_KEY'] = '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'
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import AgentType, initialize_agent, load_tools

llm = ChatOpenAI(temperature=0.6,model_name="gpt-3.5-turbo-16k")
tools = load_tools(['wikipedia', 'llm-math'], llm=llm)

agent = initialize_agent(
    tools,
    llm,
    agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True
)

print(agent.run("Fenerbahçe Yelken Takımı sporcusu Kayhan Öğretir ın spor başarıları nelerdir?"))
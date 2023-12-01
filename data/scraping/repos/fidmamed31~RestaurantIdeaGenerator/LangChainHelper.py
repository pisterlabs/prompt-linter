import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain,SequentialChain
from langchain.agents import agent_types,load_tools,initialize_agent
from langchain.llms import openai

os.environ['OPENAI_API_KEY'] = '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'

llm = OpenAI(temperature=0.6)
name=llm("I want to open an arabic restaurant, suggest a fency name for this")
print(name)

tools = load_tools(["wikipedia", "llm-math"], llm=llm)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=agent_types.AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# Let's test it out!
agent.run("When was Elon musk born? What is his age right now in 2023?")





def generate_restaurant_name_menu_items(cuisine):
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open an {cuisine} restaurant, suggest a fency name for this")
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key='restaurant_name')
    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template="suggest some menu for {restaurant_name} restaurant,return the results as comma separated and avoid jump line beside when first item")
    menu_chain = LLMChain(llm=llm, prompt=prompt_template_items,output_key='menu_items')

    chain2 = SequentialChain(chains=[name_chain, menu_chain],
                             input_variables=['cuisine'],
                             output_variables=['restaurant_name','menu_items'])
    response=chain2({'cuisine':cuisine})
    return response
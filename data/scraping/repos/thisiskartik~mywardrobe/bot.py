import json
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.schema.messages import messages_from_dict, messages_to_dict
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from .tools import search_flipkart, generate_image_caption

load_dotenv()


def bot(message, memory=None):
    system_prompt = SystemMessage(content="You are a professional fashion designer, your task it to talk to customers and generated outfit recommendations for them. You've access to search_flipkart function which can search Flipkart (India's largest ecommerce store) for products. You can then caption image url returned from these product using generate_image_caption function to further analyze the products. You should should consider factors such as the user's body type, occasion (e.g., casual, formal, party), and regional and age preferences (Ex. Young 20 year old woman looking for a Diwali outfit in Mumbai should be different to 35 year old woman in Muzzafarpur looking for a Karwa Chauth outfit). to offer appropriate and versatile outfit suggestions. You should always output the recommended products with Product URL.")
    customer_prompt = HumanMessagePromptTemplate.from_template("{text}")
    prompt_template = ChatPromptTemplate.from_messages([system_prompt, MessagesPlaceholder(variable_name="chat_history"), customer_prompt])

    if memory:
        memory = ConversationBufferMemory(chat_memory=ChatMessageHistory(messages=messages_from_dict(json.loads(memory))))
    else:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chat_model = ChatOpenAI(model_name='gpt-3.5-turbo-16k-0613')
    tools = [StructuredTool.from_function(search_flipkart), StructuredTool.from_function(generate_image_caption)]
    chain = initialize_agent(tools=tools, agent=AgentType.OPENAI_FUNCTIONS, llm=chat_model, prompt=prompt_template, verbose=True, memory=memory)

    return [chain.run(message), json.dumps(messages_to_dict(chain.memory.chat_memory.messages))]

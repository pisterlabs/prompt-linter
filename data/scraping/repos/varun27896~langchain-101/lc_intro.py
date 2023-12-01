# import os


# print("langchain era")
# os.environ["OPENAI_API_KEY"]

# # using llms
# # from langchain.llms import OpenAI

# # llm = OpenAI(temperature=0.9)
# # print(llm.predict('suggest three funky yet meaningful names that are combinations of english word and a random language word.'))

# # using chat models
# # from langchain.llms import OpenAI
# # from langchain.chat_models import ChatOpenAI
# # from langchain.schema import AIMessage, HumanMessage, SystemMessage

# # chat_model_prompt = 'what are the top 3 dangerous animals on the planet'
# # chat = ChatOpenAI(temperature=0.5)
# # print(chat.predict_messages(
# # [HumanMessage(content=chat_model_prompt)]))
# # content='The top three dangerous animals on the planet, based on their potential to cause harm to humans, are:\n\n1. Mosquitoes: Mosquitoes are responsible for transmitting deadly diseases such as malaria, dengue fever, Zika virus, and West Nile virus. They are estimated to cause millions of deaths each year, making them the most dangerous animal on Earth.\n\n2. Box jellyfish: Box jellyfish, particularly the species Chironex fleckeri and Irukandji, are known for their potent venom. Their stings can cause excruciating pain, cardiac arrest, and respiratory failure, leading to death in some cases.\n\n3. Saltwater crocodile: The saltwater crocodile, also known as the estuarine or Indo-Pacific crocodile, is the largest living reptile and is found in coastal regions of Southeast Asia and Australia. They are responsible for numerous fatal attacks on humans, as they are powerful predators and have a high tolerance for saltwater, allowing them to venture into the ocean.' additional_kwargs={} example=False
# # [AIMessage(content=chat_model_prompt)]))
# # content='The top 3 dangerous animals on the planet, based on their potential to harm humans, are:\n\n1. Mosquitoes: Mosquitoes are responsible for transmitting diseases like malaria, dengue fever, and Zika virus, causing millions of deaths worldwide each year.\n\n2. Box jellyfish: Box jellyfish are highly venomous and possess tentacles that can cause severe pain, paralysis, and even death. They are found in the waters of the Pacific and Indian Oceans.\n\n3. Saltwater crocodile: Saltwater crocodiles are the largest living reptiles and are known for their aggressive behavior. They are responsible for numerous attacks on humans, particularly in Australia and Southeast Asia.' additional_kwargs={} example=False
# # [SystemMessage(content=chat_model_prompt)]))
# # content='The top 3 dangerous animals on the planet, based on their potential threat to humans, are:\n\n1. Mosquitoes: Mosquitoes are responsible for the most significant number of human deaths worldwide. They transmit diseases such as malaria, dengue fever, Zika virus, and several other deadly illnesses.\n\n2. Box Jellyfish: Box jellyfish, particularly the species Chironex fleckeri and Carukia barnesi, are considered one of the most venomous creatures on Earth. Their tentacles contain extremely potent toxins that can cause cardiac arrest and death within minutes.\n\n3. Saltwater Crocodile: The saltwater crocodile, also known as the estuarine crocodile, holds the title for the largest living reptile and is an apex predator. They are responsible for numerous human fatalities each year, especially in regions where they coexist with humans, such as Australia and Southeast Asia.' additional_kwargs={} example=False

# # using prompt templates
# # from langchain.prompts import PromptTemplate

# # prompt = PromptTemplate.from_template('what are the different types of {name}')
# # print(prompt.format(name='predators'))

# # using chat model prompt templates
# # from langchain.prompts.chat import (
# #     HumanMessagePromptTemplate,
# #     SystemMessagePromptTemplate,
# #     ChatPromptTemplate,
# # )
# # from langchain import LLMChain
# # from langchain.chat_models import ChatOpenAI

# # chat = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo")

# # template = "You are a helpful assistant that does market research for a {type_of_company} for its {action}"
# # system_message_prompt = SystemMessagePromptTemplate.from_template(template)

# # human_prompt = "{text}"
# # human_message_prompt = HumanMessagePromptTemplate.from_template(human_prompt)

# # chat_message_prompt = ChatPromptTemplate.from_messages(
# #     [system_message_prompt, human_message_prompt]
# # )

# # # print(
# # #     chat_message_prompt.format(
# # #         type_of_company="consumer tech",
# # #         action="launch",
# # #         text="Give me the top 3 advantages of having an ai powered consumer tech product",
# # #     )
# # # )

# # chain = LLMChain(llm=chat, prompt=chat_message_prompt)
# # print(
# #     chain.run(
# #         type_of_company="consumer tech",
# #         action="launch",
# #         text="Give me the top 3 advantages of having an ai powered consumer tech product",
# #     )
# # )


# # using agents and chains
# from langchain.agents import AgentType
# from langchain.agents import load_tools
# from langchain.agents import initialize_agent
# from langchain.prompts.chat import (
#     HumanMessagePromptTemplate,
#     AIMessagePromptTemplate,
#     SystemMessagePromptTemplate,
#     ChatPromptTemplate
# )

# from langchain.chat_models import ChatOpenAI
# from langchain.llms import OpenAI

# from langchain.chains import LLMChain

# chat = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
# llm = OpenAI(temperature=0.9, model_name="gpt-3.5-turbo")

# system_template = 'You are an agent that is going to help me with your llm knowledge. You dont have to use the internet'
# system_prompt = SystemMessagePromptTemplate.from_template(system_template)

# human_template = 'what is Christopher Nolans highest grossing movie of all time?'
# human_prompt = HumanMessagePromptTemplate.from_template(human_template)

# messages = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

# chain = LLMChain(llm=chat, prompt=messages)
# print(chain.run({}))

# # tools = load_tools(["llm-math"], llm=llm)

# # agent = initialize_agent(
# #     tools, chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
# # )
# # print(agent.run(input="who was the president of india in 2003"))

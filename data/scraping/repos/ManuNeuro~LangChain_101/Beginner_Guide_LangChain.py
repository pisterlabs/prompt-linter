# 1
# from langchain.llms import OpenAI
# llm = OpenAI(temperature=0)
# prompt = "Tell me a poor joke"
# print(llm(prompt))

# # 2 - TOO SLOW!!!!!
# from langchain import HuggingFaceHub
# llm=HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0,"max_length":64})
# prompt = "why is gravity lower on the moon compared to earth?"
# print(llm(prompt))

# # 3
# from langchain.llms import OpenAI
# from langchain import PromptTemplate
# llm = OpenAI(temperature=0)

# template = "Write a {adjective} poem about {subject}"
# prompt = PromptTemplate(
#     input_variables=["adjective", "subject"],
#     template=template
# )
# result = llm(prompt.format(adjective="sad", subject="cats"))
# print(result)

# # 4 - Few Shot Examples
# from langchain.llms import OpenAI
# from langchain import PromptTemplate
# llm = OpenAI(temperature=0)

# template = """
# I want you to act as a a naming consultant for a brewery"

# Here are some good names:
# - Wadworth
# - Ushers
# - Arkells
# - Courage

# what is a good one word name for one that makes {product}
# """

# prompt = PromptTemplate(
#     input_variables=["product"],
#     template=template
# )
# result = llm(prompt.format(product="Beer"))
# print(result)

# # 5 
# from langchain.llms import OpenAI
# from langchain import PromptTemplate
# from langchain.chains import LLMChain
# llm = OpenAI(temperature=0.7)

# prompt = PromptTemplate(
#     input_variables=["product"],
#     template="what is a good name for a company that makes {product}?"
# )

# chain = LLMChain(llm=llm, prompt=prompt)
# result = chain.run("Strong Beer")
# print(result)

# Agents and Tools
# from langchain.llms import OpenAI
# from langchain.agents import load_tools
# from langchain.agents import initialize_agent

# llm = OpenAI(temperature=0.7)
# tools = load_tools(["wikipedia", "llm-math"], llm=llm)

# agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
# agent.run("What year did Nigel Mansell win the F1 world championship, what is his current age raised to the power of 0.5?")

# # Memory
# from langchain import OpenAI, ConversationChain
# llm = OpenAI(temperature=0)
# conversation = ConversationChain(llm=llm, verbose=True)

# print(conversation.predict(input="Hi There"))
# print(conversation.predict(input="Let's talk about how physics work on the moon"))
# print(conversation.predict(input="Why the gravitational field lower?"))

# # Data - Enbeddings: Numerical representation | Text Splitters | Vectorstores

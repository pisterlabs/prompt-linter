# from langchain.chat_models import ChatOpenAI
# from langchain.schema import (
#     AIMessage,
#     HumanMessage,
#     SystemMessage
# )

# chat = ChatOpenAI(openai_api_key="DONT COMMMIT THE KEY", openai_organization="org-0lR0mqZeR2oqqwVbRyeMhmrC", temperature=0)
# chat.predict_messages([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
# >> AIMessage(content="J'aime programmer.", additional_kwargs={})

# llm = OpenAI(openai_api_key="DONT COMMMIT THE KEY", openai_organization="org-0lR0mqZeR2oqqwVbRyeMhmrC", temperature=0.9)

# response = llm.predict("What would be a good company name for a company that makes colorful socks?")

# print(response)
# >> Feetful of Fun



from langchain import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="s3nh/llama2_7b_chat_uncensored-GGML",
    task="text-generation",
    model_kwargs={"temperature": 0, "max_length": 64},
)



from langchain import PromptTemplate, LLMChain

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What is electroencephalography?"

print(llm_chain.run(question))
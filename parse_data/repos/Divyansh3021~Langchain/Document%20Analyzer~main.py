import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm=HuggingFaceHub(repo_id="IlyaGusev/saiga_mistral_7b_gguf")

question = "When was Google founded?"

# print(LLMChain.run(question))

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=ConversationBufferMemory()
)

conversation.predict(input="Hi there!")
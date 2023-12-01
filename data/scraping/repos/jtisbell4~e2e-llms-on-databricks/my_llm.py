from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

DEFAULT_TEMPLATE = """
    Current conversation:
    {history}
    Human: {input}
    AI:"""


def get_llm_chain(llm) -> LLMChain:
    prompt = PromptTemplate(
        input_variables=["history", "input"], template=DEFAULT_TEMPLATE
    )

    return ConversationChain(
        llm=llm, verbose=False, prompt=prompt, memory=ConversationBufferMemory()
    )

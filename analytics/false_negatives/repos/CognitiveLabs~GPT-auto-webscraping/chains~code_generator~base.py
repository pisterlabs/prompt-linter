from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from chains.code_generator.templates import chat_script_prompt


def chain_code_generator(llm) -> LLMChain:
    # Memory
    script_memory = ConversationBufferMemory(
        input_key="output_format", memory_key="chat_history"
    )

    # Chain
    return LLMChain(
        llm=llm,
        prompt=chat_script_prompt,
        verbose=True,
        output_key="script",
        memory=script_memory,
    )

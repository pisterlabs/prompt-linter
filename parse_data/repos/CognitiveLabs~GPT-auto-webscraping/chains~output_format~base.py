from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from chains.output_format.templates import output_format_chat_prompt


def chain_output_format(llm) -> LLMChain:
    # memory
    html_memory = ConversationBufferMemory(
        input_key="html_content", memory_key="chat_history"
    )

    # chain
    return LLMChain(
        llm=llm,
        prompt=output_format_chat_prompt,
        verbose=True,
        output_key="output_format",
        memory=html_memory,
    )

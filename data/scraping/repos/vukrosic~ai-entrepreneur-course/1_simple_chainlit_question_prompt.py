import chainlit as cl
from langchain import PromptTemplate, OpenAI, LLMChain


template = """Question: {question}

Answer: Let's think step by step."""

@cl.on_chat_start
def main():
    prompt = PromptTemplate(template=template, input_variables = ["question"])
    llm_chain = LLMChain(
        prompt = prompt,
        llm=OpenAI(
            temperature=1, 
            openai_api_key="YOUR_OPENAI_API_KEY", 
            streaming=True 
        ),
        verbose=True)

    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def main(message : str):
    llm_chain = cl.user_session.get("llm_chain")

    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    await cl.Message(content=res["text"]).send()
import logging
import chainlit as cl
from langchain.llms import Ollama
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from jira import create_jira_ticket

from jira_constants import *
from chainlit.input_widget import Select

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)
logger = logging.getLogger(__name__)
llm = Ollama(
    base_url="http://127.0.0.1:11434",
    model="llama2:13b",
    temperature=0,
)


@cl.on_chat_start
async def main():
    # Instantiate the chain for that user session
    settings = await cl.ChatSettings(
        [
            Select(
                id="Task",
                label="Select Task",
                values=["jira", "coding"],
                initial_index=0,
            )
        ]
    ).send()
    value = settings["Task"]
    print("ASJDASJIDJASJD", value)
    if value == "jira":
        print("ASJDASJIDJASJD", value)
    prompt = PromptTemplate(template=template,
                            input_variables=[
                                "project_idea",
                            ],
                            partial_variables={"description": JIRA_DESCRIPTION},
                            )

    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    res = await cl.make_async(llm_chain)(
        message, callbacks=[cl.LangchainCallbackHandler()]
    )

    print(res)

    extracted_array = extract_code(res.get("text"))

    print(extracted_array, "mytexttt")

    # chain_output = llm_chain.predict(project_idea=res)

    # print(chain_output, )

    # await cl.Message(
    #     content=chain_output,
    # ).send()

    create_jira_ticket(extracted_array)

    await cl.Message(content=res["text"]).send()
    return llm_chain


def extract_code(text):
    # Splitting based on triple backticks
    if "```" in text:
        blocks = text.split("```")
        
        # Filtering out empty blocks and taking every alternate block starting from the second one, which should contain the code
        code_blocks = [block.strip() for block in blocks if block.strip()][1::2]
        
        # Joining the blocks while ignoring the lines with backticks
        return "\n".join([line for block in code_blocks for line in block.splitlines() if not line.strip().startswith("```")])
    
    # Splitting based on triple dashes
    if "---" in text:
        blocks = text.split("---")
        
        # Filtering out empty blocks and taking every alternate block starting from the second one, which should contain the code
        code_blocks = [block.strip() for block in blocks if block.strip()][1::2]
        
        # Joining the blocks while ignoring the lines with dashes
        return "\n".join([line for block in code_blocks for line in block.splitlines() if not line.strip().startswith("---")])
    
    # Splitting based on <code></code> tags
    if "<code>" in text and "</code>" in text:
        blocks = text.split("<code>")
        end_blocks = [block.split("</code>")[0] for block in blocks if "</code>" in block]
        
        # Joining the blocks while ignoring the lines with code tags
        return "\n".join([line.strip() for block in end_blocks for line in block.splitlines() if not (line.strip().startswith("<code>") or line.strip().startswith("</code>"))])
    
    return text
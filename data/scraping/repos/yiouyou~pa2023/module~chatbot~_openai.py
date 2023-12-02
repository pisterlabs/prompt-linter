# from langchain.agents import Tool
# from langchain.agents import AgentType
# from langchain.memory import ConversationBufferMemory
# from langchain.chat_models import ChatOpenAI
# from langchain.agents import initialize_agent
# from langchain.callbacks import get_openai_callback


def convert_to_md(text):
    import re
    text = text.replace("$","&#36;")
    def replace_leading_tabs_and_spaces(line):
        new_line = []
        
        for char in line:
            if char == "\t":
                new_line.append("&#9;")
            elif char == " ":
                new_line.append("&nbsp;")
            else:
                break
        return "".join(new_line) + line[len(new_line):]
    markdown_text = ""
    lines = text.split("\n")
    in_code_block = False
    for line in lines:
        if in_code_block is False and line.startswith("```"):
            in_code_block = True
            markdown_text += f"{line}\n"
        elif in_code_block is True and line.startswith("```"):
            in_code_block = False
            markdown_text += f"{line}\n"
        elif in_code_block:
            markdown_text += f"{line}\n"
        else:
            line = replace_leading_tabs_and_spaces(line)
            line = re.sub(r"^(#)", r"\\\1", line)
            markdown_text += f"{line}  \n"
    return markdown_text

def format_chat_history(_memory):
    from langchain.schema import HumanMessage
    updated_conversation = '<div style="background-color: hsl(30, 100%, 30%); color: white; padding: 5px; margin-bottom: 10px; text-align: center; font-size: 1.5em;">Chat History</div>'
    for i, message in enumerate(_memory.chat_memory.messages):
        if isinstance(message, HumanMessage):
            prefix = "User: "
            background_color = "hsl(0, 0%, 40%)"  # Dark grey background
            text_color = "hsl(0, 0%, 100%)"  # White text
        else:
            prefix = "Chatbot: "
            background_color = "hsl(0, 0%, 95%)"  # White background
            text_color = "hsl(0, 0%, 0%)"  # Black text
        updated_conversation += f'<div style="color: {text_color}; background-color: {background_color}; margin: 5px; padding: 5px;">{prefix}{message.content}</div>'
    return updated_conversation


class ChatAgent:
    def __init__(self, memory, chain):
        self.memory = memory
        self.chain = chain

def create_chatopenai(seed_memory=None):
    from langchain.agents import AgentType
    from langchain.chat_models import ChatOpenAI
    from langchain.memory import ConversationBufferMemory
    from langchain.agents import initialize_agent
    from module.tools import tools_faiss_azure_googleserp_math
    memory = seed_memory if seed_memory is not None else ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    import os
    _chatopenai = ChatOpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0)
    chain = initialize_agent(
        tools_faiss_azure_googleserp_math,
        _chatopenai,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True,
        memory=memory
    )
    return ChatAgent(memory, chain)

def predict(chatagent_openai, _ask, _chatbot, _history):
    from langchain.callbacks import get_openai_callback
    _memory = chatagent_openai.memory
    chatagent_openai = create_chatopenai(seed_memory=_memory)
    if _ask=="":
        yield _chatbot, _history, "🚩 Empty Context"
        return
    if len(_ask) > 2048:
        yield _chatbot, _history, "🚩 Input Too Long"
        return
    with get_openai_callback() as cb:
        _ans = chatagent_openai.chain.run(_ask).strip()
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
        print(_token_cost)
        # print(f"memory: {_memory}")
        # print(f"memory.chat_memory.messages: {_memory.chat_memory.messages}")
    if _ans:        
        _history = _history + [[_ask, _ans]]
        _chatbot = [[y[0], convert_to_md(y[1])] for y in _history]
        # _chatbot_html = format_chat_history(chatagent_openai.memory)
    try:
        yield _chatbot, _history, f"🚩 Generate: Success {_token_cost}"
    except:
        pass

def retry_bot(chatagent_openai, _ask, _chatbot, _history):
    if len(_history) == 0:
        yield _chatbot, _history, "🚩 Empty Context"
        return
    else:
        _chatbot.pop()
        _ask = _history.pop()[0]
        for x in predict(chatagent_openai, _ask, _chatbot, _history):
            yield x


def chat_predict_langchain(message, history):
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import AIMessage, HumanMessage
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    llm_chat = ChatOpenAI(
        temperature=0,
        model='gpt-3.5-turbo',
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm_chat(history_langchain_format)
    return gpt_response.content

def chat_predict_openai(message, history):
    import openai
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages= history_openai_format,         
        temperature=0,
        stream=True
    )
    partial_message = ""
    for chunk in response:
        if len(chunk['choices'][0]['delta']) != 0:
            partial_message = partial_message + chunk['choices'][0]['delta']['content']
            yield partial_message


def listen_cmd(cmd): # cmd = 'python', '-m' 'your_langchain.py'
    import subprocess
    """from http://blog.kagesenshi.org/2008/02/teeing-python-subprocesspopen-output.html
    """
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = []
    while True:
        line = p.stdout.readline()
        stdout.append(line)
        # print(line)
        if line == '' and p.poll() != None:
            break
    return ''.join(stdout)


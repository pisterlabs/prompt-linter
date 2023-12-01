f"""
uploaded_file = st.file_uploader("{title}", type={data_type}, key='{variable}')
        """f"""
if uploaded_file is not None:
    # Create a temporary file to store the uploaded content
    extension = uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{{extension}}') as temp_file:
        temp_file.write(uploaded_file.read())
        {variable} = temp_file.name # it shows the file path
else:
    {variable} = ''
        """f"""
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  
        st.markdown(message["content"])
        
if {variable} := st.chat_input("{placeholder}"):
    with st.chat_message("user"):
        st.markdown({variable})
    st.session_state.messages.append({{"role": "user", "content": {variable}}})
        """f"""
with st.chat_message("assistant"):
    message_placeholder = st.empty()
    full_response = ""
    # Simulate stream of response with milliseconds delay
    for chunk in {res}.split():
        full_response += chunk + " "
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    if full_response:
        st.session_state.messages.append({{"role": "assistant", "content": full_response}})        
        """f"""
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.tools import DuckDuckGoSearchRun
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.agents.tools import Tool
from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)
        """f"""
def {function_name}({argument}):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", openai_api_key=openai_api_key)
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    tools = [
        DuckDuckGoSearchRun(name="Search"),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        )]
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    return executor({argument}, callbacks=[st_cb])["output"]
        """"""
if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    {variable} = ""
elif {argument}:
    {variable} = {function_name}({argument})
else:
    {variable} = ''
        """f"""
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chains import LLMMathChain
from langchain.callbacks import StreamlitCallbackHandler
        """f"""
def {function_name}({argument}):
    search_input = "{res}".format({argument}={argument})
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    tools = [
        DuckDuckGoSearchRun(name="Search"),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        ),
    ]
    model = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    return agent.run(search_input, callbacks=[st_cb])
        """f"""
if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    {variable} = ""
elif {argument}:
    {variable} = {function_name}({argument})
else:
    {variable} = ''
        """f"""if os.path.exists('Notion_DB') and os.path.isdir('Notion_DB'):
            shutil.rmtree('Notion_DB')
        os.system(f"unzip {{{argument}}} -d Notion_DB")
        loader = {loader}("Notion_DB")"""f"""
import shutil
from langchain.document_loaders import *

        """f"""

def {function_name}({argument}):
    {loader_line}
    docs = loader.load()
    return docs
        """f"""
if {argument}:
    {variable} = {function_name}({argument})
else:
    {variable} = ''
        """f"""
from langchain.docstore.document import Document
        """f"""
{variable} =  [Document(page_content={argument}, metadata={{'source': 'local'}})]
        """f"""
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
        """f"""
def {function_name}({argument}):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", openai_api_key=openai_api_key)
    chain = load_summarize_chain(llm, chain_type="stuff")
    with st.spinner('DemoGPT is working on it. It might take 5-10 seconds...'):
        return chain.run({argument})
        """f"""
if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    {variable} = ""
elif {argument}:
    {variable} = {function_name}({argument})
else:
    variable = ""
""""""
You are a head of engineering team that gives plan to the developer to write application code.
You will see the Client's Message. The developer only does what you say nad he doesn't know Client's Message.
The plan should be broken down into clear, logical steps that detail how to develop the application. 
Consider all necessary user interactions, system processes, and validations, 
and ensure that the steps are in a logical sequence that corresponds to the given Client's Message.
Don't generate impossible steps in the plan because only those tasks are available:
{TASK_DESCRIPTIONS}

Pay attention to the input_data_type and the output_data_type.
If one of the task's output is  input of another, then output_data_type of previous one
should be the same as input_data_type of successor.

Only those task types are allowed to be used:
{TASK_NAMES}

Highly pay attention to the input data type and the output data type of the tasks while creating the plan. These are the data types:

{TASK_DTYPES}

When you create a step in the plan, its input data type 
either should be none or the output data type of the caller step. 

If you use a task in a step, highly pay attention to the input data type and the output data type of the task because it should be compatible with the step.

{helper}
""""""
Don't generate redundant steps which is not meant in the instruction.
For chat-based inputs, use "ui_input_chat" and chat-based outputs use "ui_output_chat"
Keep in mind that you cannot use python task just after plan_and_execute task. 

{helper}

Client's Message: Application that can analyze the user
System Inputs: []
Let’s think step by step.
1. Generate question to understand the personality of the user by [prompt_template() ---> question]
2. Show the question to the user [ui_output_text(question)]
3. Get answer from the user for the asked question by [ui_input_text(question) ---> answer]
4. Analyze user's answer by [prompt_template(question,answer) ---> analyze]
5. Show the result to the user by [ui_output_text(analyze)].

Client's Message: Create a system that can summarize a powerpoint file
System Inputs:[powerpoint_file]
Let’s think step by step.
1. Get file path from the user for the powerpoint file [ui_input_file() ---> file_path]
2. Load the powerpoint file as Document from the file path [doc_loader(file_path) ---> file_doc]
3. Generate summarization from the Document [doc_summarizer(file_doc) ---> summarized_text] 
5. If summarization is ready, display it to the user [ui_output_text(summarized_text)]

Client's Message: Create a translator app which translates to any language
System Inputs:[output_language, source_text]
Let’s think step by step.
1. Get output language from the user [ui_input_text() ---> output_language]
2. Get source text which will be translated from the user [ui_input_text() ---> source_text]
3. If all the inputs are filled, translate text to output language [prompt_template(output_language, source_text) ---> translated_text]
4. If translated text is ready, show it to the user [ui_output_text(translated_text)]

Client's Message: Generate a system that can generate tweet from hashtags and give a score for the tweet.
System Inputs:[hashtags]
Let’s think step by step.
1. Get hashtags from the user [ui_input_text() ---> hashtags]
2. If hashtags are filled, create the tweet [prompt_template(hashtags) ---> tweet]
3. If tweet is created, generate a score from the tweet [prompt_template(tweet) ---> score]
4. If score is created, display tweet and score to the user [ui_output_text(score)]

Client's Message: Create an app that enable me to make conversation with a mathematician 
System Inputs:[text]
Let’s think step by step.
1. Get message from the user [ui_input_chat() ---> text] 
2. Generate the response coming from the mathematician [chat(text) ---> mathematician_response]
3. If response is ready, display it to the user with chat interface [ui_output_chat(mathematician_response)]

Client's Message: Summarize a text taken from the user
System Inputs:[text]
Let’s think step by step.
1. Get text from the user [ui_input_text() ---> text] 
2. Summarize the given text [prompt_template(text) ---> summarized_text]
3. If summarization is ready, display it to the user [ui_output_text(summarized_text)]

Client's Message: Create a system that can generate blog post related to a website
System Inputs: [url]
Let’s think step by step.
1. Get website URL from the user [ui_input_text() ---> url]
2. Load the website as Document from URL [doc_loader(url) ---> web_doc]
3. Convert Document to string content [doc_to_string(web_doc) ---> web_str ]
4. If string content is generated, generate a blog post related to that string content [prompt_template(web_str) ---> blog_post]
5. If blog post is generated, display it to the user [ui_output_text(blog_post)]

Client's Message: {instruction}
System Inputs:{system_inputs}
Let’s think step by step.
"""f"""
if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    {variable} = ""
elif {" and ".join(list(map(inputs_joiner,inputs)))}:
    with st.spinner('DemoGPT is working on it. It takes less than 10 seconds...'):
        {variable} = {signature}
else:
    {variable} = ""
"""f"""
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

msgs = StreamlitChatMessageHistory()

def {signature}:
    prompt = PromptTemplate(
        input_variables={input_variables}, template='''{system_template}'''
    )
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="{human_input}", chat_memory=msgs, return_messages=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", openai_api_key=openai_api_key, temperature={temperature})
    chat_llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False,
        memory=memory
        )
    
    return chat_llm_chain.run({run_call})
    
{function_call} 

    """f"""
if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    {variable} = ""
elif {" and ".join(list(map(inputs_joiner,inputs)))}:
    with st.spinner('DemoGPT is working on it. It takes less than 10 seconds...'):
        {variable} = {signature}
else:
    {variable} = ""
            """f"""
if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    {variable} = ""
else:
    with st.spinner('DemoGPT is working on it. It takes less than 10 seconds...'):
        {variable} = {signature}
            """f"""
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

    """"""
msgs = StreamlitChatMessageHistory()
    """f"""
def {signature}:
    prompt = PromptTemplate(
        input_variables={input_variables}, template='''{system_template}'''
    )
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="{human_input}", chat_memory=msgs, return_messages=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", openai_api_key=openai_api_key, temperature={temperature})
    chat_llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False,
        memory=memory
        )
    
    return chat_llm_chain.run({run_call})
    """f"""
if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    {variable} = ""
elif {" and ".join(list(map(inputs_joiner,inputs)))}:
    with st.spinner('DemoGPT is working on it. It takes less than 10 seconds...'):
        {variable} = {signature}
else:
    {variable} = ""
            """f"""
if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    {variable} = ""
else:
    with st.spinner('DemoGPT is working on it. It takes less than 10 seconds...'):
        {variable} = {signature}
            """f"""\n
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)

def {signature}:
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        openai_api_key=openai_api_key,
        temperature={temperature}
    )
    system_template = \"\"\"{templates['system_template']}\"\"\"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = \"\"\"{templates['template']}\"\"\"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run({run_call})
    return result # returns string   

{function_call}               

"""f"""
if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    {variable} = ""
else:
    with st.spinner('DemoGPT is working on it. It takes less than 10 seconds...'):
        {variable} = {signature}
        """f"""
if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    {variable} = ""
elif {" and ".join(list(map(inputs_joiner,inputs)))}:
    with st.spinner('DemoGPT is working on it. It takes less than 10 seconds...'):
        {variable} = {signature}
else:
    {variable} = ""
            """f"""
if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    {variable} = ""
else:
    with st.spinner('DemoGPT is working on it. It takes less than 10 seconds...'):
        {variable} = {signature}
            """f"""\n
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)
    """f"""

def {signature}:
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        openai_api_key=openai_api_key,
        temperature={temperature}
    )
    system_template = \"\"\"{templates['system_template']}\"\"\"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = \"\"\"{templates['template']}\"\"\"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run({run_call})
    return result # returns string   

""""""
import os
import streamlit as st
import tempfile
""""""
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    placeholder="sk-...",
    value=os.getenv("OPENAI_API_KEY", ""),
    type="password",
)
"""f"""
uploaded_file = st.file_uploader("{title}", type={data_type}, key='{variable}')
if uploaded_file is not None:
    # Create a temporary file to store the uploaded content
    extension = uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{{extension}}') as temp_file:
        temp_file.write(uploaded_file.read())
        {variable} = temp_file.name # it shows the file path
else:
    {variable} = ''
        """f"""
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  
        st.markdown(message["content"])
        
if {variable} := st.chat_input("{placeholder}"):
    with st.chat_message("user"):
        st.markdown({variable})
    st.session_state.messages.append({{"role": "user", "content": {variable}}})
        """f"""
import time

with st.chat_message("assistant"):
    message_placeholder = st.empty()
    full_response = ""
    # Simulate stream of response with milliseconds delay
    for chunk in {res}.split():
        full_response += chunk + " "
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    if full_response:
        st.session_state.messages.append({{"role": "assistant", "content": full_response}})        
        """f"""
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.tools import DuckDuckGoSearchRun
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.agents.tools import Tool
from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)

def {function_name}({argument}):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", openai_api_key=openai_api_key)
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    tools = [
        DuckDuckGoSearchRun(name="Search"),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        )]
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    return executor({argument}, callbacks=[st_cb])["output"]

if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    {variable} = ""
elif {argument}:
    {variable} = {function_name}({argument})
else:
    {variable} = ''
  
        """f"""
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chains import LLMMathChain
from langchain.callbacks import StreamlitCallbackHandler

def {function_name}({argument}):
    search_input = "{res}".format({argument}={argument})
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    tools = [
        DuckDuckGoSearchRun(name="Search"),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        ),
    ]
    model = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    return agent.run(search_input, callbacks=[st_cb])
        
if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    {variable} = ""
elif {argument}:
    {variable} = {function_name}({argument})
else:
    {variable} = ''
        """f"""if os.path.exists('Notion_DB') and os.path.isdir('Notion_DB'):
            shutil.rmtree('Notion_DB')
        os.system(f"unzip {{{argument}}} -d Notion_DB")
        loader = {loader}("Notion_DB")"""f"""
import shutil
from langchain.document_loaders import *

def {function_name}({argument}):
    {loader_line}
    docs = loader.load()
    return docs
if {argument}:
    {variable} = {function_name}({argument})
else:
    {variable} = ''
        """f"""
from langchain.docstore.document import Document
{variable} =  [Document(page_content={argument}, metadata={{'source': 'local'}})]
        """f"""
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

def {function_name}({argument}):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", openai_api_key=openai_api_key)
    chain = load_summarize_chain(llm, chain_type="stuff")
    with st.spinner('DemoGPT is working on it. It might take 5-10 seconds...'):
        return chain.run({argument})
if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
    {variable} = ""
elif {argument}:
    {variable} = {function_name}({argument})
else:
    {variable} = ""
""""""create a game where the system creates a story and stops at the exciting point and asks to the user 
    to make a selection then after user makes his selection, system continues to the story depending on the user's selection""""""
Generate a prompt to guide the model in executing specific role. It acts as directives, providing the context and structure needed for the model to respond appropriately.

Components:
1. "system_template": Describes the model's role and task for a given instruction. This string will be used with system_template.format(...) so only used curly braces for inputs
2. "human_input": It is one of the input keys from the "Inputs" list. It should be the most appropriate one that you think it is coming from chat input. 
2. "variety": Indicates how creative or deterministic the model's response should be.
3. "function_name": A unique identifier for the specific task or instruction.

IMPORTANT NOTE:
- Write "system_template" in a way that, system_template.format(input=something for input in inputs) work.
It should also have {{chat_history}}
What I mean is that, put all the elements of Inputs inside of system_template with curly braces so that I can format it with predefined parameters.
Always put the most similar variable name which should be coming from chat input in curly braces at the end .
It should be strictly a JSON format so that it can be directly used by json.loads function in Python.
""""""
IMPORTANT NOTE:
- ONLY the variables listed under "Inputs" MUST be included in either the "system_template" section within curly braces (e.g., '{{variable_name}}'). Do NOT include any other parameters within curly braces.
- Ensure that the exact variable names listed in "Inputs" are used without any modifications.
- If a variable is listed in "Inputs," it must appear within curly braces in the "system_template".
=========================================
Instruction: Generate a blog post from a title.
Inputs: ["human_input","title"]
Args: {{
"system_template":"
You are a chatbot having a conversation with a human. You are supposed to write a blog post from given title. Human want you to generate a blog post but you are also open to feedback and according to the given feedback, you can refine the blog \n\nTitle:{{title}}\n\n{{chat_history}}\nHuman: {{human_input}}\nBlogger:",
"human_input":"human_input",
"variety": "True",
"function_name": "chat_blogger"
}}
##########################################
Instruction: Generate a response in the style of a psychologist with a given tone.
Inputs: ["talk_input","tone"]
Args: {{
"system_template": "You are a psychologist. Reply to your patience with the given tone\n\nTone:{{tone}}\n\n{{chat_history}}\nPatience: {{talk_input}}\nPsychologist:",
"human_input":"talk_input",
"variety": "False",
"function_name": "talk_like_a_psychologist"
}}
##########################################
Instruction: Answer question related to the uploaded powerpoint file.
Inputs: ["question","powerpoint_doc"]
Args: {{
"system_template": "You are a chatbot having a conversation with a human.\n\nGiven the following extracted parts of a long document, chat history and a question, create a final answer.\n\n{{powerpoint_doc}}\n\n{{chat_history}}\nHuman: {{question}}\nChatbot:",
"human_input":"question",
"variety": "False",
"function_name": "talk_like_a_psychologist"
}}
##########################################
Instruction: Generate answer similar to a mathematician
Inputs: ["human_input"]
Args: {{
"system_template": "You are a mathematician. Solve the human's mathematics problem as efficient as possible.\n\n{{chat_history}}\nHuman: {{human_input}}\nMathematician:",
"human_input":"human_input",
"variety": "True",
"function_name": "solveMathProblem"
}}
##########################################
Instruction:{instruction}
Inputs:{inputs}
Args:
""""""
Create a Python list of task objects that align with the provided instruction and plan. Task objects must be Python dictionaries, and the output should strictly conform to a Python list of JSON objects.

You must use only the tasks provided in the description:

{TASK_DESCRIPTIONS}

task_name could be only one of the task names below:
{TASK_NAMES}
""""""
Create a Python list of task objects that align with the provided instruction and all steps of the plan.

Task objects must be Python dictionaries, and the output should strictly conform to a Python list of JSON objects.

Follow these detailed guidelines:

Task Objects: Create a Python dictionary for each task using the following keys:

step: It represents the step number corresponding to which plan step it matches
task_type: Should match one of the task names provided in task descriptions.
task_name: Define a specific name for the task that aligns with the corresponding plan step.
input_key: List the "output_key" values from parent tasks used as input or "none" if there's no input or if it comes from the user.
input_data_type: The list of data types of the inputs
output_key: Designate a unique key for the task's output. It is compatible with the output type if not none
output_data_type: The data type of the output
description: Provide a brief description of the task's goal, mirroring the plan step.

Ensure that each task corresponds to each step in the plan, and that no step in the plan is omitted.
Ensure that output_key is unique for each task.
Ensure that each task corresponds to each step in the plan
Ensure that an output type of task does not change.

##########################
Instruction: Create a system that can analyze the user
Plan:
Let’s think step by step.
1. Generate question to understand the personality of the user by 'prompt_template'
2. Show the question to the user with 'ui_output_text'
3. Get answer from the user for the asked question with 'ui_input_text'
4. Analyze user's answer by 'prompt_template'.
5. Show the analyze to the user with 'ui_output_text'
List of Task Objects (Python List of JSON):
[
    {{
        "step": 1,
        "task_type": "prompt_template",
        "task_name": "generate_question",
        "input_key": "none",
        "input_data_type": "none",
        "output_key": "question",
        "output_data_type": "string",
        "description": "Generate question to understand the personality of the user"
    }},
    {{
        "step": 2,
        "task_type": "ui_output_text",
        "task_name": "show_question",
        "input_key": "question",
        "input_data_type": "string",
        "output_key": "none",
        "output_data_type": "none",
        "description": "Display the AI-generated question to the user."
    }},
    {{
        "step": 3,
        "task_type": "ui_input_text",
        "task_name": "get_answer",
        "input_key": "none",
        "input_data_type": "none",
        "output_key": "answer",
        "output_data_type": "string",
        "description": "Ask the user to input the answer for the generated question"
    }},
    {{
        "step": 4,
        "task_type": "prompt_template",
        "task_name": "analyze_answer",
        "input_key": ["question", "answer"],
        "input_data_type": ["string","string"],
        "output_key": "prediction",
        "output_data_type": "string",
        "description": "Predict horoscope of the user given the question and user's answer to that question"
    }},
    {{
        "step": 5,
        "task_type": "ui_output_text",
        "task_name": "show_analyze",
        "input_key": "prediction",
        "input_data_type": "string",
        "output_key": "none",
        "output_data_type": "none",
        "description": "Display the AI's horoscope prediction"
    }}
]
##########################
Instruction: Create a system that can generate blog post related to a website
Plan:
1. Get website URL from the user with 'ui_input_text'
2. Use 'doc_loader' to load the page as Document
3. Use 'doc_to_string' to convert Document to string
4. Use 'prompt_template' to generate a blog post using the result of doc_to_string
5. If blog post is generated, show it to the user with 'ui_output_text'.
List of Task Objects (Python List of JSON):
[
    {{
        "step": 1,
        "task_type": "ui_input_text",
        "task_name": "get_url",
        "input_key": "none",
        "input_data_type": "none",
        "output_key": "url",
        "output_data_type": "string",
        "description": "Get website url from the user"
    }},
    {{
        "step": 2,
        "task_type": "doc_loader",
        "task_name": "doc_loader",
        "input_key": "url",
        "input_data_type": "string",
        "output_key": "docs",
        "output_data_type": "Document",
        "description": "Load the document from the website url"
    }},
    {{
        "step": 3,
        "task_type": "doc_to_string",
        "task_name": "convertDocToString",
        "input_key": "docs",
        "input_data_type": "Document",
        "output_key": "docs_string",
        "output_data_type": "string",
        "description": "Convert docs to string"
    }},
    {{
        "step": 4,
        "task_type": "prompt_template",
        "task_name": "writeBlogPost",
        "input_key": ["docs_string"],
        "input_data_type": ["string"],
        "output_key": "blog",
        "output_data_type": "string",
        "description": "Write blog post related to the context of docs_string"
    }},
    {{
        "step": 5,
        "task_type": "ui_output_text",
        "task_name": "show_blog",
        "input_key": "blog",
        "input_data_type": "string",
        "output_key": "none",
        "output_data_type": "none",
        "description": "Display the generated blog post to the user"
    }}
]
##########################
Instruction: Summarize uploaded file and convert it to language that user gave.
Plan:
1. Get file path using 'ui_input_file'
2. Use 'ui_input_text' to get the output language from the user
3. Use 'doc_loader' to load the file as Document from file path
4. Use 'summarize' to summarize the Document
5. Use 'prompt_template' to translate the summarization
6. If translation is ready, show it to the user with 'ui_output_text'.
List of Task Objects (Python List of JSON):
[
    {{
        "step": 1,
        "task_type": "ui_input_file",
        "task_name": "get_path",
        "input_key": "none",
        "input_data_type": "none",
        "output_key": "file_path",
        "output_data_type": "string",
        "description": "Get path of the file that the user upload"
    }},
    {{
        "step": 2,
        "task_type": "ui_input_text",
        "task_name": "get_language",
        "input_key": "none",
        "input_data_type": "none",
        "output_key": "language",
        "output_data_type": "string",
        "description": "Get output language for translation"
    }},
    {{
        "step": 3,
        "task_type": "doc_loader",
        "task_name": "doc_loader",
        "input_key": "file_path",
        "input_data_type": "string",
        "output_key": "docs",
        "output_data_type": "Document",
        "description": "Load the document from the given path"
    }},
    {{
        "step": 4,
        "task_type": "summarize",
        "task_name": "summarizeDoc",
        "input_key": "docs",
        "input_data_type": "Document",
        "output_key": "summarization_result",
        "output_data_type": "string",
        "description": "Summarize the document"
    }},
    {{
        "step": 5,
        "task_type": "prompt_template",
        "task_name": "translate",
        "input_key": ["summarization_result","language"],
        "input_data_type": ["string","string"],
        "output_key": "translation",
        "output_data_type": "string",
        "description": "Translate the document into the given language"
    }},
    {{
        "step": 6,
        "task_type": "ui_output_text",
        "task_name": "show_translation",
        "input_key": "translation",
        "input_data_type": "string",
        "output_key": "none",
        "output_data_type": "none",
        "description": "Display the file summary translation to the user"
    }}
]
##########################
Instruction:{instruction}
Plan : {plan}
List of Task Objects (Python List of JSON):
""""""
Create a plan to fulfill the given instruction. 
The plan should be broken down into clear, logical steps that detail how to accomplish the task. 
Consider all necessary user interactions, system processes, and validations, 
and ensure that the steps are in a logical sequence that corresponds to the given instruction.
Don't generate impossible steps in the plan because only those tasks are available:
{TASK_DESCRIPTIONS}

Pay attention to the input_data_type and the output_data_type.
If one of the task's output is  input of another, then output_data_type of previous one
should be the same as input_data_type of successor.

Only those task types are allowed to be used:
{TASK_NAMES}

Highly pay attention to the input data type and the output data type of the tasks while creating the plan. These are the data types:

{TASK_DTYPES}

When you create a step in the plan, its input data type 
either should be none or the output data type of the caller step. 

If you use a task in a step, highly pay attention to the input data type and the output data type of the task because it should be compatible with the step.

""""""
Don't generate redundant steps which is not meant in the instruction.


Instruction: Application that can analyze the user
System Inputs: []
Let's work this out in a step by step way to be sure we have the right answer.
1. Generate question to understand the personality of the user by 'prompt_template'
2. Show the question to the user by 'ui_output_text'
3. Get answer from the user for the asked question by 'ui_input_text'
4. Analyze user's answer by 'prompt_template'.
5. Show the result to the user by 'ui_input_text'.

Instruction: Create a system that can summarize a powerpoint file
System Inputs:[powerpoint_file]
Let's work this out in a step by step way to be sure we have the right answer.
1. Get file path from the user by 'ui_input_file' for the powerpoint file
2. Use 'doc_loader' to load the powerpoint file as Document from the file path.
3. Use 'doc_summarizer' to generate summarization from the Document. 
5. If summarization is ready, display it to the user by 'ui_output_text'.

Instruction: Create a translator which translates to any language
System Inputs:[output_language, source_text]
Let's work this out in a step by step way to be sure we have the right answer.
1. Get output language from the user by 'ui_input_text'
2. Get source text which will be translated from the user by 'ui_input_text'
3. If all the inputs are filled, use 'prompt_template' to translate text to output language
4. If translated text is ready, show it to the user by 'ui_output_text'

Instruction: Generate a system that can generate tweet from hashtags and give a score for the tweet.
System Inputs:[hashtags]
Let's work this out in a step by step way to be sure we have the right answer.
1. Get hashtags from the user by 'ui_input_text'
2. If hashtags are filled, use 'prompt_template' to create tweet.
3. If tweet is created, use 'prompt_template' to generate a score from the tweet.
4. If score is created, display tweet and score to the user by 'ui_output_text'.

Instruction: Summarize a text taken from the user
System Inputs:[text]
Let's work this out in a step by step way to be sure we have the right answer.
1. Get text from the user by 'ui_input_text' 
2. Use 'prompt_template' to summarize the given text.
3. If summarization is ready, display it to the user by 'ui_output_text'.

Instruction: Create a platform which lets the user select a lecture and then show topics for that lecture 
then give a question to the user. After user gives his/her answer, it gives a score for the answer and give explanation.
System Inputs:[lecture, topic, user_answer]
Let's work this out in a step by step way to be sure we have the right answer.
1. Use 'prompt_template' to generate lectures
2. Among those generated by prompt_template, get lecture from the user by 'ui_input_text'.
3. After user selects a lecture, generate topics releated to that lecture by 'prompt_template'.
4. Among those generated by prompt_template, get topic from the user by 'ui_input_text' .
5. After user selects the topic, use 'prompt_template' to generate a question related to that topic and lecture
6. Get answer from the user by 'ui_input_text'.
7. Use 'prompt_template' to generate the real answer and score for the user's answer.
8. Display real and answer and score for the user's answer by 'ui_output_text'.

Instruction: Create a system that can generate blog post related to a website
System Inputs: [url]
Let's work this out in a step by step way to be sure we have the right answer.
1. Get website URL from the user by 'ui_input_text'
2. Use 'doc_loader' to load the website as Document from URL
3. Use 'doc_to_string' to convert Document to string content
4. If string content is generated, use 'prompt_template' to generate a blog post related to that string content.
5. If blog post is generated, display it to the user by 'ui_output_text'.

Instruction: {instruction}
Let's work this out in a step by step way to be sure we have the right answer.
""""""
Create a Python list of task objects that align with the provided instruction and plan. Task objects must be Python dictionaries, and the output should strictly conform to a Python list of JSON objects.

You must use only the tasks provided in the description:

{TASK_DESCRIPTIONS}

task_name could be only one of the task names below:
{TASK_NAMES}
""""""
Create a Python list of task objects that align with the provided instruction and all steps of the plan.

Task objects must be Python dictionaries, and the output should strictly conform to a Python list of JSON objects.

Follow these detailed guidelines:

Task Objects: Create a Python dictionary for each task using the following keys:

step: It represents the step number corresponding to which plan step it matches
task_type: Should match one of the task names provided in task descriptions.
task_name: Define a specific name for the task that aligns with the corresponding plan step.
input_key: List the "output_key" values from parent tasks used as input or "none" if there's no input or if it comes from the user.
input_data_type: The list of data types of the inputs
output_key: Designate a unique key for the task's output. It is compatible with the output type if not none
output_data_type: The data type of the output
description: Provide a brief description of the task's goal, mirroring the plan step.

Ensure that each task corresponds to each step in the plan, and that no step in the plan is omitted.
Ensure that output_key is unique for each task.
Ensure that each task corresponds to each step in the plan
Ensure that an output type of task does not change.

##########################
Instruction: Create a system that can generate blog post related to a website
Plan:
1. Get website URL from the user with 'ui_input_text'
2. Use 'doc_loader' to load the page as Document
3. Use 'doc_to_string' to convert Document to string
4. Use 'prompt_template' to generate a blog post using the result of doc_to_string
5. If blog post is generated, show it to the user with 'ui_output_text'.
List of Task Objects (Python List of JSON):
[
    {{
        "step": 1,
        "task_type": "ui_input_text",
        "task_name": "get_url",
        "input_key": "none",
        "input_data_type": "none",
        "output_key": "url",
        "output_data_type": "string",
        "description": "Get website url from the user"
    }},
    {{
        "step": 2,
        "task_type": "doc_loader",
        "task_name": "doc_loader",
        "input_key": "url",
        "input_data_type": "string",
        "output_key": "docs",
        "output_data_type": "Document",
        "description": "Load the document from the website url"
    }},
    {{
        "step": 3,
        "task_type": "doc_to_string",
        "task_name": "convertDocToString",
        "input_key": "docs",
        "input_data_type": "Document",
        "output_key": "docs_string",
        "output_data_type": "string",
        "description": "Convert docs to string"
    }},
    {{
        "step": 4,
        "task_type": "prompt_template",
        "task_name": "writeBlogPost",
        "input_key": ["docs_string"],
        "input_data_type": ["string"],
        "output_key": "blog",
        "output_data_type": "string",
        "description": "Write blog post related to the context of docs_string"
    }},
    {{
        "step": 5,
        "task_type": "ui_output_text",
        "task_name": "show_blog",
        "input_key": "blog",
        "input_data_type": "string",
        "output_key": "none",
        "output_data_type": "none",
        "description": "Display the generated blog post to the user"
    }}
]
##########################
Instruction:{instruction}
Plan : {plan}
List of Task Objects (Python List of JSON):
""""""
You are an AI assistant that write a concise prompt to direct an assistant to make web search for the given instruction.
You will have inputs and instruction. The prompt should be formattable with the inputs which means it should include inputs with curly braces.
""""""
Instruction: Search the given input
Inputs:input
Prompt: Find the answer of it: {{input}}

Instruction: Find the list of song releated to the title
Inputs:title
Prompt: Find the list of songs releated to the title: {{title}}

Instruction:{instruction}
Inputs:{inputs}
Prompt:
"""f"""
Create a plan to fulfill the given instruction. 
The plan should be broken down into clear, logical steps that detail how to accomplish the task. 
Consider all necessary user interactions, system processes, and validations, 
and ensure that the steps are in a logical sequence that corresponds to the given instruction.
Don't generate impossible steps in the plan because only those tasks are available:
{TASK_DESCRIPTIONS}

Pay attention to the input_data_type and the output_data_type.
If one of the task's output is  input of another, then output_data_type of previous one
should be the same as input_data_type of successor.

Only those task types are allowed to be used:
{TASK_NAMES}

Highly pay attention to the input data type and the output data type of the tasks while creating the plan. These are the data types:

{TASK_DTYPES}

When you create a step in the plan, its input data type 
either should be none or the output data type of the caller step. 

If you use a task in a step, highly pay attention to the input data type and the output data type of the task because it should be compatible with the step.

""""""
Don't generate redundant steps which is not meant in the instruction.


Instruction: Application that can analyze the user
System Inputs: []
Let’s think step by step.
1. Generate question to understand the personality of the user by [prompt_template() ---> question]
2. Show the question to the user [ui_output_text(question)]
3. Get answer from the user for the asked question by [ui_input_text(question) ---> answer]
4. Analyze user's answer by [prompt_template(question,answer) ---> analyze]
5. Show the result to the user by [ui_output_text(analyze)].

Instruction: Create a system that can summarize a powerpoint file
System Inputs:[powerpoint_file]
Let’s think step by step.
1. Get file path from the user for the powerpoint file [ui_input_file() ---> file_path]
2. Load the powerpoint file as Document from the file path [doc_loader(file_path) ---> file_doc]
3. Generate summarization from the Document [doc_summarizer(file_doc) ---> summarized_text] 
5. If summarization is ready, display it to the user [ui_output_text(summarized_text)]

Instruction: Create a translator which translates to any language
System Inputs:[output_language, source_text]
Let’s think step by step.
1. Get output language from the user [ui_input_text() ---> output_language]
2. Get source text which will be translated from the user [ui_input_text() ---> source_text]
3. If all the inputs are filled, use translate text to output language [prompt_template(output_language, source_text) ---> translated_text]
4. If translated text is ready, show it to the user [ui_output_text(translated_text)]

Instruction: Generate a system that can generate tweet from hashtags and give a score for the tweet.
System Inputs:[hashtags]
Let’s think step by step.
1. Get hashtags from the user [ui_input_text() ---> hashtags]
2. If hashtags are filled, create the tweet [prompt_template(hashtags) ---> tweet]
3. If tweet is created, generate a score from the tweet [prompt_template(tweet) ---> score]
4. If score is created, display tweet and score to the user [ui_output_text(score)]

Instruction: Summarize a text taken from the user
System Inputs:[text]
Let’s think step by step.
1. Get text from the user [ui_input_text() ---> text] 
2. Summarize the given text [prompt_template(text) ---> summarized_text]
3. If summarization is ready, display it to the user [ui_output_text(summarized_text)]

Instruction: Create a system that can generate blog post related to a website
System Inputs: [url]
Let’s think step by step.
1. Get website URL from the user [ui_input_text() ---> url]
2. Load the website as Document from URL [doc_loader(url) ---> web_doc]
3. Convert Document to string content [doc_to_string(web_doc) ---> web_str ]
4. If string content is generated, generate a blog post related to that string content [prompt_template(web_str) ---> blog_post]
5. If blog post is generated, display it to the user [ui_output_text(blog_post)]

Instruction: {instruction}
System Inputs:{system_inputs}
Let’s think step by step.
"""
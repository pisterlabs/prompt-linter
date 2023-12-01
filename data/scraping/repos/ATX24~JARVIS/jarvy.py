import os

#Set API keys

def setKeys():
    os.environ["OPENAI_API_KEY"] = ""


def getllm():
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    return llm
   

def getcheapllm():


    from langchain.llms import GPT4All
    from langchain import PromptTemplate, LLMChain
    llm = GPT4All()
    return llm






def getDuckDuckGoTool():
    from langchain.tools import DuckDuckGoSearchRun, Tool
    #DuckDuckGo Tool
    search = DuckDuckGoSearchRun()

    search_tool = Tool(
        name="HelperTool",
        func=search.run,
        description="Useful when needed to search the web or get the weather",
    )
    return search_tool



#GMAIL Tool
def getGmailTool():
    from langchain.agents.agent_toolkits import GmailToolkit
    toolkit = GmailToolkit()

    # tools = toolkit.get_tools()
    # print(tools)

    from langchain.tools.gmail.utils import build_resource_service, get_gmail_credentials

    # Can review scopes here https://developers.google.com/gmail/api/auth/scopes
    # For instance, readonly scope is 'https://www.googleapis.com/auth/gmail.readonly'
    credentials = get_gmail_credentials(
        token_file="token.json",
        scopes=["https://mail.google.com/"],
        client_secrets_file="credentials.json",
    )
    api_resource = build_resource_service(credentials=credentials)
    toolkit = GmailToolkit(api_resource=api_resource)
    gmail_tools = toolkit.get_tools()
    return gmail_tools


#Create IFTTT tool
def getLightsTool():
    from langchain.tools.ifttt import IFTTTWebhook

    key = ""


    onurl = f"https://maker.ifttt.com/trigger/lightson/with/key/{key}"
    offurl = f"https://maker.ifttt.com/trigger/lightsoff/with/key/{key}"

    light_tool_on = IFTTTWebhook(
        name="lightson", description="Use for turning on lights", url=onurl
    )

    light_tool_off = IFTTTWebhook(
        name="lightsoff", description="Use for turning off lights", url=offurl
    )

    return [light_tool_on, light_tool_off]











#Get other tools
#Python_repl: Can execute code
#Terminal: Ability to work with files
#Arxiv: Gets scientific papers



def getTools():
    from langchain.agents import load_tools
    tools = load_tools(["python_repl", "terminal", "arxiv"]) 
    search_tool = getDuckDuckGoTool()
    gmail_tools = getGmailTool()
    lights_tools = getLightsTool()
    for tool in lights_tools:
        tools.append(tool)

    tools.append(search_tool)
    for tool in gmail_tools:
        tools.append(tool)
    
    return tools










#Build Agent
def buildJarvis():
    
    tools = getTools()
    # llm - getllm()
    llm = getllm()
    #Memory
    from langchain.prompts import MessagesPlaceholder
    from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }

    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    #Create agent
    from langchain.agents import initialize_agent, Tool
    from langchain.agents import AgentType

    #Use structured zero shot reaction in the future
    jarvis = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True, agent_kwargs=agent_kwargs,
        memory=memory,)


    return jarvis

def buildcheapJarvis():

    tools = getTools()
    # llm - getllm()
    llm = getcheapllm()
    #Memory
    from langchain.prompts import MessagesPlaceholder
    from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }

    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    #Create agent
    from langchain.agents import initialize_agent, Tool
    from langchain.agents import AgentType

    jarvis = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True, agent_kwargs=agent_kwargs,
        memory=memory,)
  
    return jarvis


def get_audio():
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = ""
        try:
            said = r.recognize_google(audio)
        except Exception as e:
            print(str(e))
    return said



    




def runJarvis():
    setKeys()
    jarvis = buildJarvis()
    response = jarvis.run("From now on you are to take the role of a helper assistant named Friday. My name is Tony Stark.")
    print(response)

    while True:
        print("Stark...")
        q1 = get_audio()
        if ('stop listening' in q1):
            print('ok')
            break
        if ('friday' in q1 or "Friday" in q1):
            #Speech response
            from gtts import gTTS
            import os
            language = 'en'
            myobj = gTTS(text="on it", lang=language, tld='co.uk', slow=False)
            myobj.save('response.mp3')
            os.system("mpg123 response.mp3")
            os.system("rm response.mp3")
            
            response = jarvis.run(q1)
            myobj = gTTS(text=response, lang=language, tld='co.uk', slow=False)
            myobj.save('response.mp3')
            os.system("mpg123 response.mp3")
            os.system("rm response.mp3")
       
def runcheapJarvis():
    setKeys()
    jarvis = buildcheapJarvis()
    

def cheapTest():
    from langchain import PromptTemplate, LLMChain
    from langchain.llms import GPT4All
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])

    local_path = (
    "./models/ggml-gpt4all-l13b-snoozy.bin"  # replace with your desired local file path
    )

    import requests

    from pathlib import Path
    from tqdm import tqdm

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    # Example model. Check https://github.com/nomic-ai/gpt4all for the latest models.
    url = 'http://gpt4all.io/models/ggml-gpt4all-l13b-snoozy.bin'

    # send a GET request to the URL to download the file. Stream since it's large
    response = requests.get(url, stream=True)

    # open the file in binary mode and write the contents of the response to it in chunks
    # This is a large file, so be prepared to wait.
    with open(local_path, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192)):
            if chunk:
                f.write(chunk)

    callbacks = [StreamingStdOutCallbackHandler()]

    # Verbose is required to pass to the callback manager
    llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)

    # If you want to use a custom model add the backend parameter
    # Check https://docs.gpt4all.io/gpt4all_python.html for supported backends
    llm = GPT4All(model=local_path, backend="gptj", callbacks=callbacks, verbose=True)

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

    llm_chain.run(question)

    




runJarvis()
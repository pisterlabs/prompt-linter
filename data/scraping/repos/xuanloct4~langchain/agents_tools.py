
import environment


def DocstoreExplorer_search_tool(store=None):
    from langchain import Wikipedia
    from langchain.agents import initialize_agent, Tool
    from langchain.agents import AgentType
    from langchain.agents.react.base import DocstoreExplorer
    defaultStore = Wikipedia()
    if store is None:
        store = defaultStore
    docstore=DocstoreExplorer(store)
    tool = Tool(
            name="Search",
            func=docstore.search,
            description="useful for when you need to ask with search"
        )
    agentType=AgentType.REACT_DOCSTORE
    return tool, agentType

def DocstoreExplorer_lookup_tool(store=None):
    from langchain import Wikipedia
    from langchain.agents import initialize_agent, Tool
    from langchain.agents import AgentType
    from langchain.agents.react.base import DocstoreExplorer
    defaultStore = Wikipedia()
    if store is None:
        store = defaultStore
    docstore=DocstoreExplorer(store)
    tool = Tool(
            name="Lookup",
            func=docstore.lookup,
            description="useful for when you need to ask with lookup"
        )
    agentType=AgentType.REACT_DOCSTORE
    return tool, agentType

def chinook_db_tool(llm):
    from langchain import SQLDatabase, SQLDatabaseChain
    from langchain.agents import Tool
    db = SQLDatabase.from_uri("sqlite:///./Chinook.db")
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    tool = Tool(
        name="FooBar DB",
        func=db_chain.run,
        description="useful for when you need to answer questions about FooBar. Input should be in the form of a question containing full context"
    )
    return tool

def calculator_tool(llm):
    from langchain import LLMMathChain
    from langchain.agents import Tool
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)
    tool=Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    )
    return tool

def search_tool_serpapi(name=None):
    from langchain.agents import Tool
    from langchain import SerpAPIWrapper
    defaultName = "Search"
    if name is None:
        name = defaultName
    search = SerpAPIWrapper()
    tool=Tool(name = name,
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world. the input to this should be a single search term.",
            return_direct=True
        )
    return tool

def gradio_tools_StableDiffusionTool(prompt):
    # prompt = "Please create a photo of a dog riding a skateboard"
    from gradio_tools.tools import StableDiffusionTool
    local_file_path = StableDiffusionTool().langchain.run(prompt)
    print(local_file_path)

def gradio_tools_multipleTools():
    from langchain.agents import initialize_agent, AgentType

    from gradio_tools.tools import (StableDiffusionTool,
                                    ImageCaptioningTool,
                                    StableDiffusionPromptGeneratorTool,
                                    TextToVideoTool)

    tools = [StableDiffusionTool().langchain,
            ImageCaptioningTool().langchain,
            StableDiffusionPromptGeneratorTool().langchain,
            TextToVideoTool().langchain]
    agentType = AgentType.CONVERSATIONAL_REACT_DESCRIPTION

    # from langchain.memory import ConversationBufferMemory
    # memory = ConversationBufferMemory(memory_key="chat_history")
    # agent = initialize_agent(tools, llm, memory=memory, agent="conversational-react-description", verbose=True)
    # output = agent.run(input=("Please create a photo of a dog riding a skateboard "
    #                         "but improve my prompt prior to using an image generator."
    #                         "Please caption the generated image and create a video for it using the improved prompt."))
    # print(output)
    return tools, agentType

def multiplierTool():
    from langchain.agents import initialize_agent, AgentType, Tool
    # from langchain import OpenAI
    # llm = OpenAI(temperature=0)
    from langchain.tools import StructuredTool

    def multiplier(a: float, b: float) -> float:
        """Multiply the provided floats."""
        return a * b

    def parsing_multiplier(string):
        a, b = string.split(",")
        return multiplier(int(a), int(b))

    tool = StructuredTool.from_function(multiplier)
    # Structured tools are compatible with the STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION agent type.
    agentType = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
    ###Or use with string format

    # tool = Tool(
    #         name = "Multiplier",
    #         func=parsing_multiplier,
    #         description="useful for when you need to multiply two numbers together. The input to this tool should be a comma separated list of numbers of length two, representing the two numbers you want to multiply together. For example, `1,2` would be the input if you wanted to multiply 1 by 2."
    #     )
    # agentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION
    return tool, agentType


def human_input_tool(llm = None):
    from langchain.agents import load_tools
    from langchain.agents import AgentType

    # tools = load_tools(["human", "llm-math"],llm=llm)
    # agentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION

    def get_input() -> str:
        print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
        contents = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == "q":
                break
            contents.append(line)
        return "\n".join(contents)


    # You can modify the tool when loading
    tools = load_tools(
        ["human", "ddg-search"],
        llm=llm,
        input_func=get_input
    )
    # # Or you can directly instantiate the tool
    # from langchain.tools import HumanInputRun
    # tool = HumanInputRun(input_func=get_input)
    # tools = [tool]

    agentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION

    # agent_chain = initialize_agent(
    #     tools,
    #     llm,
    #     agent=agentType,
    #     verbose=True,
    # )
    # agent_chain.run("I need help attributing a quote")
    # agent_chain.run("What's my friend Eric's surname?")
    
    return tools, agentType
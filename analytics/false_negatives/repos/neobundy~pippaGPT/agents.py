from langchain.chat_models import ChatOpenAI
from langchain import Wikipedia
from dotenv import load_dotenv
from langchain.chains import LLMMathChain, LLMChain
from langchain.agents import Tool, initialize_agent, load_tools
from langchain.prompts import PromptTemplate
from langchain.agents.react.base import DocstoreExplorer
from langchain import SerpAPIWrapper
import os
import settings


def get_agent_llm():
    load_dotenv()
    return ChatOpenAI(temperature=settings.DEFAULT_GPT_AGENT_HELPER_MODEL_TEMPERATURE,
                      model_name=settings.DEFAULT_GPT_AGENT_HELPER_MODEL)


def math_tool(llm, tools):
    llm_math = LLMMathChain.from_llm(llm)
    tool = Tool(
        name='Calculator',
        func=llm_math.run,
        description='Useful for answering math questions.'
    )
    tools.append(tool)
    return tools


def llm_tool(llm, tools):
    prompt = PromptTemplate(
        input_variables=["query"],
        template="{query}"
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool = Tool(
        name='Generic Language Model',
        func=llm_chain.run,
        description='Use this tool for general queries and logic.'
    )
    tools.append(tool)
    return tools


def wiki_tool(tools):
    docstore = DocstoreExplorer(Wikipedia())
    my_tools = [
        Tool(
            name="Search",
            func=docstore.search,
            description='Search Wikipedia'
        ),
        Tool(
            name="Lookup",
            func=docstore.lookup,
            description='Look up a term in Wikipedia'
        )
    ]
    tools.extend(my_tools)
    return tools


# An agent method should follow the naming convention: get_<agent_name_prefix>_agent
# The agent name prefix should be the same as the agent name in the settings.py file

def get_math_agent(llm, memory=None):
    tools = []
    tools = math_tool(llm, tools)
    tools = llm_tool(llm, tools)

    return initialize_agent(
        agent="zero-shot-react-description",
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=settings.MAX_AGENTS_ITERATIONS,
        memory=memory,
        handle_parsing_errors=True,
    )


def get_wiki_agent(llm, memory=None):
    tools = []
    tools = wiki_tool(tools)
    return initialize_agent(
        tools,
        llm,
        agent="react-docstore",
        verbose=True,
        max_iterations=settings.MAX_AGENTS_ITERATIONS,
        memory=memory,
        handle_parsing_errors="Check your output and make sure it conforms!",
    )


def get_google_agent(llm, memory=None):
    search = SerpAPIWrapper(serpapi_api_key=os.environ['SERPAPI_API_KEY'])

    tools = [
        Tool(
            name="Intermediate Answer",
            func=search.run,
            description='Google Search'
        )
    ]

    return initialize_agent(
        tools,
        llm,
        agent="self-ask-with-search",
        verbose=True,
        memory=memory,
        max_iterations=settings.MAX_AGENTS_ITERATIONS,
        handle_parsing_errors="Check your output and make sure it conforms!",
    )


def get_dalle_agent(llm, memory=None):
    tools = load_tools(['dalle-image-generator'])
    return initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True,
        max_iterations=settings.MAX_AGENTS_ITERATIONS,
        memory=memory,
        handle_parsing_errors="Check your output and make sure it conforms!",
    )

def get_midjourney_agent(llm, memory=None):
    basic_midjourney_prompt_template = """
You are an expert at generating image generative ai tool midjourney prompts. You always follow the guidelines:

/imagine prompt: [art style or cinematic style] of [subject], [in the style of or directed  by] [artist or director], [scene], [lighting], [colors], [composition], [focal length], [f-stop], [ISO]

[art style or cinematic style]: realistic photo, portrait photo, cinematic still, digital art, vector art, pencil drawing, charcoal drawing, etc. Pick only one art style. If an art style is specified in the subject, use that style.
[subject]: the subject in the scene
 [in the style of or directed  by]: in the style of an artist or directed by a director
[scene]: describe the scene of the [subject]
[artist or director]: recommend a beffiting artist or director
[lighting]: recommend a lighting setup fitting for the scene of the [subject]
[colors]: recommend colors fitting for the scene of the [subject]
[composition]: recommend a composition such as portrait, cowboy, body shot, close-up, extreme close-up, etc., fitting for the scene of the [subject]
[focal length]: recommend a camera focal length fitting for the scene of the [subject]
[f-stop]: recommend a camera f-stop fitting for the scene of the [subject]
[ISO]: recommend an ISO value fitting for the scene of the [subject]; include the word "ISO"

Create a mid-journey prompt following the above guidelines. Insert the generated prompt into a Python code snippet:

```python

[generated midjourney prompt] --s 750 --q 1 --ar 2:1 --seed [random number ranging from 0 to 4294967295]

```

Examples:

Human: cinematic still of a strikingly beautiful female warrior

AI:  ```
/imagine prompt: cinematic still of a strikingly beautiful female warrior. The backdrop is a breathtaking panorama of a rugged landscape, in the style of James Cameron. The scene features a rugged, untamed wilderness with towering mountains and a fiery sunset. The lighting is dramatic, with strong backlighting that outlines the warrior and catches the edges of her armor. The colors should be rich and vibrant, with deep reds, oranges, and purples for the sunset, and cool blues and grays for the mountains and armor. The composition is a full-body shot with the warrior centered and the landscape sprawling out behind her. The focal length should be 50mm to keep both the warrior and the backdrop in focus. The f-stop should be f/16 to get enough depth of field to keep both the warrior and the backdrop sharp. The ISO should be 100 to keep the image clean and free of noise. --s 750 --q 1 --ar 2:1 --seed 3742891634
```

Human: pencil drawing of a strikingly beautiful female warrior
AI: ```
/imagine prompt: pencil drawing of a strikingly beautiful female warrior... [same as the above]
```

Human: {query}
AI:
"""

    prompt = PromptTemplate(
        input_variables=["query"],
        template=basic_midjourney_prompt_template
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    return initialize_agent(
        agent="zero-shot-react-description",
        tools=[Tool(
            name='Midjourney Prompter',
            func=llm_chain.run,
            description="Use this tool to generate a midjourney generative ai image description code snippet. The first output of this tool is ALWAYS right and final. No further action should be taken."
                        "Do not modify or remove any character in the output. You must return the output exactly as it is."
                        "The final answer should be wrapped within ''' and ''' code block."
        ), ],
        llm=llm,
        verbose=True,
        max_iterations=settings.MAX_AGENTS_ITERATIONS,
        memory=memory,
        handle_parsing_errors=True,
    )


def main():
    llm = get_agent_llm()

    # my_agent = get_math_agent(llm)
    # my_agent("what is (pi * 2.5)^3.5?")
    # my_agent("what is the capital of South Korea?")
    #
    # my_agent = get_wiki_agent(llm)
    # my_agent("When did Antoni Gaudi die?")
    #
    # my_agent = get_google_agent(llm)
    # my_agent("Who is the oldest among the heads of state of South Korea, the US, and Japan?")
    # my_agent("Who gives the highest price target of Tesla in Wall Street? And what's the price target?")

    # my_agent = get_midjourney_agent(llm)
    # my_agent("cinematic still of a strikingly beautiful female teenage warrior")

    my_agent = get_dalle_agent(llm)
    my_agent("cinematic still of a strikingly beautiful female teenage warrior")

if __name__ == "__main__":
    main()
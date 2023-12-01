# ------------------------------- Import Libraries -------------------------------
from tools.tools import get_profile_url
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

# ------------------------------- Twitter Lookup Function ------------------------------- 
def lookup(name: str) -> str:
    # Initialize the ChatOpenAI model with specific parameters
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    
    # Define the prompt template for the agent's request
    template = """
       Given the name {name_of_person}, find a link to their Twitter profile page and extract their username.
       The final answer should contain only the person's username."""
    
    # Define tools available for the agent, in this case to crawl Google for Twitter profiles
    tools_for_agent_twitter = [
        Tool(
            name="Crawl Google for Twitter profile page",
            func=get_profile_url,
            description="Useful for retrieving the Twitter Page URL",
        ),
    ]
    
    # Initialize the agent with specified tools and settings
    agent = initialize_agent(
        tools_for_agent_twitter,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    
    # Create a prompt template object with the given format
    prompt_template = PromptTemplate(
        input_variables=["name_of_person"], template=template
    )
    
    # Use the agent to run the prompt and get the Twitter username
    twitter_username = agent.run(prompt_template.format_prompt(name_of_person=name))
    
    return twitter_username


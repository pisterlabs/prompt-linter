from langchain.agents import initialize_agent, Tool, AgentType

from tools.tools import google_search_2, get_website_content

def lookup(requisito: str) -> str:

    from langchain.chat_models import ChatOpenAI
    from langchain import PromptTemplate
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")

    from datetime import datetime

    # Get the current date and time
    current_datetime = datetime.now()

    # Extract the date part (today's date) from the datetime object
    today_date = current_datetime.date()

    tools_for_agent =[
        Tool(
            name="Search on Google from a string", 
            func=google_search_2,
            description="usefull for when you need to search on the web",
        ),
        Tool(
            name="Open this Website from a url", 
            func=get_website_content,
            description="usefull for when you need to see whats in a website, input starts with http",
        )
        
    ]
    
    agent = initialize_agent(
        tools=tools_for_agent, 
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True,
        handle_parsing_errors=True
        )

    summary_template = """
    Você é especialista em técnicas para buscas no google.
    Dado "{requisito}" encontre e me traga um edital público aberto sobre esse tema e informações úteis sobre esse edital.
    Faça sua pesquisa em ptbt
    Respire fundo, pense passo a passo para garantir que encontrará as respostas corretas. Vc consegue!
    """

    prompt_template = PromptTemplate(
        input_variables=["requisito"], 
        template=summary_template,
    )

    linkedin_profile_url = agent.run(
        prompt_template.format_prompt(requisito=requisito))

    return linkedin_profile_url
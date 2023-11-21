def load_csv_agent(csv_file, text, model_name="gpt-3.5-turbo", temperature=0.0):
    from langchain.agents import create_csv_agent
    from langchain.chat_models import ChatOpenAI

    model = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
    )
    langchain_agent = create_csv_agent(
        llm=model,
        path=csv_file,
    )
    langchain_agent.agent.llm_chain.prompt.template

    return langchain_agent.run(text)

def babyagi():
    import os
    from langchain.chains import LLMChain
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.experimental import BabyAGI
    from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
    from langchain import OpenAI, LLMChain
    from dotenv import load_dotenv
    load_dotenv()
    _todo_prompt = PromptTemplate.from_template(
        "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
    )
    _todo_chain = LLMChain(
        llm=OpenAI(temperature=0),
        prompt=_todo_prompt
    )
    from module.tools import tools_faiss_azure_googleserp
    _tools = tools_faiss_azure_googleserp
    _tool_todo = Tool(
        name="TODO",
        func=_todo_chain.run,
        description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
    )
    _tools.append(_tool_todo)

    _prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks:    {context}."""
    _suffix = """Question: {task}
    {agent_scratchpad}"""
    _prompt = ZeroShotAgent.create_prompt(
        _tools,
        prefix=_prefix,
        suffix=_suffix,
        input_variables=["objective", "task", "context", "agent_scratchpad"],
    )
    _llm = OpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0)
    _llm_chain = LLMChain(
        llm=_llm,
        prompt=_prompt
    )
    _tool_names = [tool.name for tool in _tools]
    _agent = ZeroShotAgent(
        llm_chain=_llm_chain,
        allowed_tools=_tool_names
    )
    _agent_executor = AgentExecutor.from_agent_and_tools(
        agent=_agent,
        tools=_tools,
        verbose=True
    )
    from langchain.vectorstores import FAISS
    from langchain.docstore import InMemoryDocstore
    from langchain.embeddings import OpenAIEmbeddings
    embeddings_model = OpenAIEmbeddings()
    import faiss
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    _vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    _babyagi = BabyAGI.from_llm(
        llm=_llm,
        vectorstore=_vectorstore,
        task_execution_chain=_agent_executor,
        max_iterations=3,
        verbose=False,
    )
    return _babyagi


def run_babyagi(_task):
    _ans, _steps = "", ""
    from langchain.callbacks import get_openai_callback
    with get_openai_callback() as cb:
        _babyagi = babyagi()
        _re = _babyagi({"objective": _task})
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
        print(_token_cost)
        print(_re)
        _ans = _re
        _steps = f"{_token_cost}\n\n"
    
    return [_ans, _steps]
    

if __name__ == "__main__":
    
    _task = "Write a weather report for SF today"
    _re1 = run_babyagi(_task)
    print(_re1)



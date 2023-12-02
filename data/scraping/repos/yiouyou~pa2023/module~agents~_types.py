def agent_react_zeroshot(_loaded_tools, _ask):
    import os
    from langchain.callbacks import get_openai_callback
    from langchain.agents import initialize_agent
    from langchain.agents import AgentType
    from langchain.llms import OpenAI
    from module.util import parse_intermediate_steps
    llm = OpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0)
    agent = initialize_agent(
        _loaded_tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=10,
        max_execution_time=20,
        early_stopping_method="force" # "force/generate"
    )
    _ans, _steps = "", ""
    with get_openai_callback() as cb:
        _re = agent({"input": _ask})
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
        print(_token_cost)
        _ans = _re["output"]
        _steps = f"{_token_cost}\n\n" + parse_intermediate_steps(_re["intermediate_steps"])
    return [_ans, _steps]

def agent_react_docstore(_loaded_tools, _ask):
    import os
    from langchain.callbacks import get_openai_callback
    from langchain.agents import initialize_agent
    from langchain.agents import AgentType
    from langchain import OpenAI
    from module.util import parse_intermediate_steps
    llm = OpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0)
    agent = initialize_agent(
        _loaded_tools,
        llm,
        agent=AgentType.REACT_DOCSTORE,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=10,
        max_execution_time=20,
        early_stopping_method="force" # "force/generate"
    )
    _ans, _steps = "", ""
    with get_openai_callback() as cb:
        _re = agent({"input": _ask})
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
        print(_token_cost)
        _ans = _re["output"]
        _steps = f"{_token_cost}\n\n" + parse_intermediate_steps(_re["intermediate_steps"])
    return [_ans, _steps]

def agent_selfask_search(_loaded_tools, _ask):
    import os
    from langchain.callbacks import get_openai_callback
    from langchain.agents import initialize_agent
    from langchain.agents import AgentType
    from langchain import OpenAI
    from module.util import parse_intermediate_steps
    llm = OpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0)
    agent = initialize_agent(
        _loaded_tools,
        llm,
        agent=AgentType.SELF_ASK_WITH_SEARCH,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=10,
        max_execution_time=20,
        early_stopping_method="force" # "force/generate"
    )
    _ans, _steps = "", ""
    with get_openai_callback() as cb:
        _re = agent({"input": _ask})
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
        print(_token_cost)
        _ans = _re["output"]
        _steps = f"{_token_cost}\n\n" + parse_intermediate_steps(_re["intermediate_steps"])
    return [_ans, _steps]

def agent_plan_execute(_loaded_tools, _ask):
    from langchain.callbacks import get_openai_callback
    from langchain.chat_models import ChatOpenAI
    from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
    import os
    model = ChatOpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0)
    planner = load_chat_planner(model)
    executor = load_agent_executor(
        model,
        _loaded_tools,
        verbose=True
    )
    agent = PlanAndExecute(
        planner=planner,
        executor=executor,
        verbose=True,
        include_run_info=True
    )
    _ans, _steps = "", ""
    with get_openai_callback() as cb:
        _re = agent.run(_ask)
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
        print(_token_cost)
        _ans = _re
        _steps = f"{_token_cost}\n\n"
    return [_ans, _steps]

def agent_openai_multifunc(_loaded_tools, _ask):
    from langchain.callbacks import get_openai_callback
    from langchain.agents import initialize_agent
    from langchain.agents import AgentType
    from langchain.chat_models import ChatOpenAI
    from module.util import parse_intermediate_steps
    import os
    llm = ChatOpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0)
    agent = initialize_agent(
        _loaded_tools,
        llm,
        agent=AgentType.OPENAI_MULTI_FUNCTIONS, # OPENAI_FUNCTIONS
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=10,
        max_execution_time=20,
        early_stopping_method="force" # "force/generate"
    )
    _ans, _steps = "", ""
    with get_openai_callback() as cb:
        _re = agent({"input": _ask})
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
        print(_token_cost)
        _ans = _re["output"]
        _steps = f"{_token_cost}\n\n" + parse_intermediate_steps(_re["intermediate_steps"])
    return [_ans, _steps]



if __name__ == "__main__":

    import sys
    from pathlib import Path
    from pprint import pprint
    # pprint(sys.path)
    _pwd = Path(__file__).absolute()
    _pa_path = _pwd.parent.parent.parent
    # print(_pa_path)
    sys.path.append(str(_pa_path))
    # pprint(sys.path)
    from dotenv import load_dotenv
    load_dotenv()
    
    from module.tools import tools_faiss_azure_googleserp_math
    from module.tools import tools_faiss_azure_langchain_googleserp_math
    from module.tools import tools_react_docstore_wiki
    from module.tools import tools_selfask_search
    from module.tools import tools_google
    
    _re1 = agent_react_zeroshot(
        tools_google,
        "How many people live in canada as of 2023?"
    )
    print(_re1)

    _re2 = agent_react_zeroshot(
        tools_faiss_azure_langchain_googleserp_math,
        "how many disk types can be used as OS disk, and what are they?"
    )
    print(_re2)

    _re3 = agent_react_docstore(
        tools_react_docstore_wiki,
        "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?"
    )
    print(_re3)

    _re4 = agent_selfask_search(
        tools_selfask_search,
        "What is the hometown of the reigning men's U.S. Open champion?"
    )
    print(_re4)

    _re5 = agent_plan_execute(
        tools_faiss_azure_googleserp_math,
        "what's the pricing structure for Premium SSD v2? please provide a formula to calculate it."
    )
    print(_re5)


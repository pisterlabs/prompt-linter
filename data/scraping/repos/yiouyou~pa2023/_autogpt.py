def autogpt():
    from langchain.tools.file_management.write import WriteFileTool
    from langchain.tools.file_management.read import ReadFileTool
    from dotenv import load_dotenv
    load_dotenv()
    from module.tools import tools_faiss_azure_googleserp
    _tools = tools_faiss_azure_googleserp
    _tools.append(WriteFileTool())
    _tools.append(ReadFileTool())
    from langchain.vectorstores import FAISS
    from langchain.docstore import InMemoryDocstore
    from langchain.embeddings import OpenAIEmbeddings
    embeddings_model = OpenAIEmbeddings()
    import faiss
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    _vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    from langchain.experimental import AutoGPT
    from langchain.chat_models import ChatOpenAI
    from langchain.memory.chat_message_histories import FileChatMessageHistory
    import os
    _llm = ChatOpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0)
    _agent = AutoGPT.from_llm_and_tools(
        ai_name="Zack",
        ai_role="Assistant",
        tools=_tools,
        llm=_llm,
        memory=_vectorstore.as_retriever(),
        chat_history_memory=FileChatMessageHistory("chat_history.txt"),
    )
    _agent.chain.verbose = True
    return _agent


def run_autogpt(_task):
    _ans, _steps = "", ""
    from langchain.callbacks import get_openai_callback
    with get_openai_callback() as cb:
        _autogpt = autogpt()
        _re = _autogpt.run([_task])
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
        print(_token_cost)
        print(_re)
        _ans = _re
        _steps = f"{_token_cost}\n\n"
    
    return [_ans, _steps]
    


if __name__ == "__main__":
    
    _task = "Write a weather report for SF today"
    _re1 = run_autogpt(_task)
    print(_re1)



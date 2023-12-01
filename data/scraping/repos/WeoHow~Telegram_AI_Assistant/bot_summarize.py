from dotenv import load_dotenv

from langchain.callbacks import get_openai_callback
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()


def summarize_docs(docs, doc_url, llm):
    print(docs)
    print (f'You have {len(docs)} document(s) in your {doc_url} data')
    print (f'There are {len(docs[0].page_content)} characters in your document')

    # 切割文檔
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=3)

    # 加載切割後的文檔
    split_docs = text_splitter.split_documents(docs)

    print (f'You have {len(split_docs)} split document(s)')

    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False)

    with get_openai_callback() as cb:
        response = chain.run(input_documents=split_docs)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")

    return response
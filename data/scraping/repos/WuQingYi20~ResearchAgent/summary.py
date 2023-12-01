from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

def summary(RAObject, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=[" ", ".", ";", ","],
        chunk_size=10000,
        chunk_overlap=500,
        length_function=len,
    )
    docs = text_splitter([content])
    map_summary = """
    write a summary of the following text for {RAObject}:
    "{text}"
    Summary:
    """
    map_prompt_template = PromptTemplate(
        template=map_summary, input_variables=["text", "RAObject"]
    )
    summary_chain = load_summarize_chain(
        llm,
        map_prompt_template,
        chian_type="map_reduce",
        verbose=True
    )

    output = summary_chain.run(input_document=docs, RAObject=RAObject)
    print("use summary funcs")
    return output

from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate


def load_chain():
    prompt_template = """You are an expert writer, your mission is order the doc without lose information.
    
    Doc: {doc}

    Format doc here:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["doc"]
    )

    chain = LLMChain(
        llm=AzureChatOpenAI(deployment_name="openai", temperature="0"),
        prompt=PROMPT,
        verbose=True
    )

    return chain


def order_doc(doc):
    chain = load_chain()
    
    chunk_size = 5000
    
    chunks = [doc[i:i + chunk_size] for i in range(0, len(doc), chunk_size)]

    results = []
    for chunk in chunks:
        output = chain.run(doc=chunk)
        results.append(output)

    final_output = "\n".join(results)
    print(final_output)
    return final_output

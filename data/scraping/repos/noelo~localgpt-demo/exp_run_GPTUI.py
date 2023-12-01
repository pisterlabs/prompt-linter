import os
import openai
import logging
# import chainlit as cl
from chainlit import AskUserMessage, Message, on_chat_start
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from InstructorEmbedding import INSTRUCTOR
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY



SYSTEM_TEMPLATE = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

```
The answer is foo
SOURCES: xyz
```

Begin!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

# @on_chat_start
# def main():
#         Message(
#             content=f"Ask questions to the OpenShift Documentation",
#         ).send()


# @cl.langchain_factory
def load_model(): 

    model_id = "TheBloke/vicuna-7B-1.1-HF"
    
    logging.info(f"Loading model.....{model_id}")
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(model_id)
    generation_config = GenerationConfig.from_pretrained(model_id)
    
    logging.info("Loading LlamaTokenizer.....")
    # tokenizer = LlamaTokenizer.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)


    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config
    )



    pipe = pipeline(
        "text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


# @cl.langchain_postprocess
def process_response(res):
    answer = res["result"]
    sources = res["source_documents"]
    source_elements = []
    found_sources = []

    # Get the metadata and texts from the user session
    # metadatas = cl.user_session.get("metadatas")
    # all_sources = [m["source"] for m in metadatas]
    # texts = cl.user_session.get("texts")

    if sources:


    #     # Add the sources to the message
        i = 0
        for source in sources:
            # print(source)
            
    #         # Get the index of the source
    #         try:
    #             index = all_sources.index(source_name)
    #         except ValueError:
    #             continue
    #         text = texts[index]
            found_sources.append(source.metadata)
    #         # Create the text element referenced in the message
            # source_elements.append(cl.Text(id=i,text=source.metadata['source'], name=source.metadata['source'], display="side"))
            i+=1

        # if found_sources:
        #     print(found_sources)
        #     answer += f"\nSources: {', '.join(found_sources)}"
        # else:
        #     answer += "\nNo sources found"
        
    x = []
    text_content = "Hello, this is a text element."
    
    for src in found_sources:
        print(src)
        src_str = src['source']
        res_str = src_str.replace("/home/noelo/dev/localGPT/SOURCE_DOCUMENTS/", "")

        # x.append(cl.Text(name=res_str, text="https://docs.openshift.com", display="inline"))

    print(source_elements)
    # cl.Message(content=answer, elements=x).send() #NOC

    # cl.Text(name="simple_text", text=text_content, display="inline").send()
    # cl.Text(name="simple_text", text="this is a test", display="inline").send()
    
def main():
    logging.info("Loading model.....")

    llm=load_model()
    embedding_function = HuggingFaceInstructEmbeddings()
# load the vectorstore
    db = Chroma(collection_name='OCP',persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_function, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        
    
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s',level=logging.DEBUG)
    main()
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from huggingface_hub import hf_hub_download
from constants import MODELS_PATH

MODEL_ID = "TheBloke/Llama-2-7b-Chat-GGUF"
MODEL_BASENAME = "llama-2-7b-chat.Q4_K_M.gguf"

def load_model(model_id=MODEL_ID, model_basename=MODEL_BASENAME):
    '''
    download model for the first run. 
    Subsequent runs will use the downloaded model.
    '''
    model_path = hf_hub_download(
            repo_id=model_id,
            filename=model_basename,
            resume_download=True,
            cache_dir=MODELS_PATH,
        )
    kwargs = {
            "model_path": model_path,
            "n_ctx": 4096,
            "max_tokens": 4096,
            "n_batch": 512,
        }

    return LlamaCpp(**kwargs)

def main():
    # load the instructorEmbeddings
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                model_kwargs={"device": "cuda"})
    # load the vectorstore
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    # load the LLM for generating Natural Language responses. 
    llm = load_model(model_id=MODEL_ID, model_basename=MODEL_BASENAME)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        
        # Get the answer from the chain
        res = qa(query)    
        answer, docs = res['result'], res['source_documents']

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)
        
        # # Print the relevant sources used for the answer
        print("----------------------------------SOURCE DOCUMENTS---------------------------")
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)
        print("----------------------------------SOURCE DOCUMENTS---------------------------")

if __name__ == "__main__":
    main()
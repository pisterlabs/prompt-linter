from langchain.chains import RetrievalQA
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY
from transformers import AutoModel,LlamaTokenizer, AutoTokenizer,LlamaForCausalLM, pipeline
import click
import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import pdb
from constants import CHROMA_SETTINGS

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    '''
    Select a model on huggingface. 
    If you are running this for the first time, it will download a model for you. 
    subsequent runs will use the model from the disk. 
    '''

    # tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int8", trust_remote_code=True, cache_dir='/persistent-storage/')
    # model = AutoModel.from_pretrained("THUDM/chatglm-6b-int8", trust_remote_code=True, cache_dir='/persistent-storage/').half().cuda()
    # model.eval()

    quantized_model_dir = r"/persistent-storage/vicuna-7B-GPTQ-4bit-128g"
    quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,
    )

    model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir,
        use_safetensors=True,
        model_basename="vicuna-7B-GPTQ-4bit-128g",
        use_triton=True,
        quantize_config=quantize_config)
    
    tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, use_fast=True)
    
    #model_id = "Thireus/Vicuna13B-v1.1-8bit-128g"
    #model_id = "TheBloke/vicuna-7B-1.1-HF"
    # tokenizer = LlamaTokenizer.from_pretrained(model_id)
    # model = LlamaForCausalLM.from_pretrained(model_id,
    #                                         load_in_8bit=True, # set these options if yousr GPU supports them!
    #                                         device_map='auto',
    #                                         torch_dtype=torch.float16,
    #                                         low_cpu_mem_usage=True,
    #                                         cache_dir = "/persistent-storage/"
    #                                         )

    pipe = pipeline(
        "text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        device=0
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)

    return local_llm

@click.command()
@click.option('--device_type', default='gpu', help='device to run on, select gpu or cpu')
def main(device_type, ):
    # load the instructorEmbeddings
    if device_type in ['cpu', 'CPU']:
        device='cpu'
    else:
        device='cuda'

    print(f"Running on: {device}")
        
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                model_kwargs={"device": device})
    # load the vectorstore
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    # Prepare the LLM
    # callbacks = [StreamingStdOutCallbackHandler()]
    # load the LLM for generating Natural Language responses. 
    llm = load_model()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    #pdb.set_trace()
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

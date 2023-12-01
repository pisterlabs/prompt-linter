from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA

def chat(chat_history, user_input):
    bot_response = qa_chain({"query": user_input})
    bot_response = bot_response['result']
    response = "".join(bot_response)
    chat_history.append((user_input, response))
    return chat_history

checkpoint = "MBZUAI/LaMini-Flan-T5-783M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Load the model with offload_folder argument
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    offload_folder="ipc_vector_data/92ca7e32-064c-43f8-973e-1060c76a995e",  # Replace with the actual path
    device_map="auto",
    torch_dtype=torch.float32
)

embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

db = Chroma(persist_directory="ipc_vector_data", embedding_function=embeddings)

pipe = pipeline(
    'text2text-generation',
    model=base_model,
    tokenizer=tokenizer,
    max_length=512,
    do_sample=True,
    temperature=0.3,
    top_p=0.95
)
local_llm = HuggingFacePipeline(pipeline=pipe)

qa_chain = RetrievalQA.from_chain_type(
    llm=local_llm,
    chain_type='stuff',
    retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
    return_source_documents=True,
)

# Example usage
chat_history = []
user_input = input("Enter your initial input: ")
chat_history = chat(chat_history, user_input)

# Display the conversation history
for prompt, response in chat_history:
    print(f"User: {prompt}")
    print(f"Saul: {response}")

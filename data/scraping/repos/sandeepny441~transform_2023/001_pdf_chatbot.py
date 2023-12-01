import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index import LlamaIndex

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("t5-base") 
tokenizer = AutoTokenizer.from_pretrained("t5-base")

# Initialize LlamaIndex and upload PDF  
docs = LlamaIndex()
with open("my_data.pdf", "rb") as pdf_file:
    pdf_contents = pdf_file.read()
docs.add_documents(pdf_contents)

# Define query and context formatting functions
def format_query(query):
    return f"chatbot: {query}"

def format_context(contexts):
    return "\n".join(f"Doc: {doc}" for doc in contexts)

# Chatbot loop
while True:
    # Get user input query
    query = input("You: ")
    
    # Retrieve relevant context from PDF using LlamaIndex
    encoded_query = tokenizer(format_query(query), return_tensors="pt")
    encoded_docs = docs.retrieve(query=encoded_query, top_k=10)
    context = format_context(encoded_docs)
    
    # Generate response 
    input_ids = tokenizer(context, return_tensors="pt").input_ids
    gen_tokens = model.generate(input_ids, max_length=512)
    response = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

    print(f"Chatbot: {response}")
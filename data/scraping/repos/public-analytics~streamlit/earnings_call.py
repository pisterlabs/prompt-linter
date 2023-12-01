import streamlit as st
import openai
import pinecone
import tiktoken
import time

#converts text to GPT4 token count. used to calculate context size
def num_tokens(string):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

#connect to pinecone
def pinecone_init(key, index):
    pinecone.init(
        api_key=key,
        environment="us-east-1-aws"
    )
    return pinecone.Index(index)

#sorts pinecone results by relevance score (desc) and id (asc) to build context string
def sort_list_by_two_values(lst):
    return sorted(lst, key=lambda d: (-d['score'], d['id']))

#convert text to vector 
def get_embed(query, model):
    openai.api_key = st.secrets["openai"]
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )
    return res['data'][0]['embedding']

#queries pinecone for k number of nearest points
def get_context(index, vector, k):
    res = index.query(vector, top_k=k, include_metadata=True)
    res_ordered = sort_list_by_two_values(res['matches'])
    return res_ordered

#combines pinecone context into token-limited sized string
def build_context(context, tokens):
    contexts = [x['metadata']['text'] for x in context]
    
    #optional prompt
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    
    # append contexts until hitting token limit
    for i in range(0, len(contexts)):
        if num_tokens("\n\n---\n\n".join(contexts[:i])) >= tokens:
            raw_context = ("\n\n---\n\n".join(contexts[:i-1])) 
            prompt = (prompt_start + raw_context + prompt_end)
            break
        elif i == len(contexts)-1:
            raw_context = ("\n\n---\n\n".join(contexts))
            prompt = (prompt_start + raw_context + prompt_end)
            
    return {'prompt':prompt, 'context':raw_context}

def chat(query_with_context, model):
    openai.api_key = st.secrets["openai"]
    res = openai.ChatCompletion.create(
      model=model,
      messages=[
            {"role": "system", "content": "You are a helpful investment bot."},
            {"role": "user", "content": query_with_context}
        ]
    )
    return res

#################################################################
###Streamlit###

st.title('Invest Bot')
st.subheader('Public.com Pinecone POC')

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    index = st.selectbox('Pinecone Index', ('earnings-call', 'public-faq'))
with col2:
    k = st.number_input('Pinecone Results', format='%i', value=10)
with col3:
    token_limit = st.number_input('Context Max Tokens', format='%i', value = 7000)
with col4:
    embed_model = st.selectbox('Embed Model', ("text-embedding-ada-002", "placeholder"))
with col5:
    model = st.selectbox('Chat Model', ('gpt-4', 'gpt-3.5-turbo'))

query = st.text_input('Enter Query', 'Who spoke at the Apple Q1 2023 earnings call')

if query:
    with st.spinner('Wait for it...'):
        vector = get_embed(query, embed_model)
        pinecone_index = pinecone_init(st.secrets["pinecone"], index)
        context = get_context(pinecone_index, vector, k)
        query_with_context = build_context(context, token_limit)['prompt']
        output = chat(query_with_context, model)
        st.success(output['choices'][0]['message']['content'].replace("$","\$"))
    
    with st.expander("View Pinecone Query Results"):
        for x in context:
            st.write(f"id: {x['id']}, title: {x['metadata']['title']}, score: {str(x['score'])}")
    
    with st.expander("View Query With Context"):
            st.write(query_with_context)

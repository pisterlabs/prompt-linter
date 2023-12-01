import streamlit as st
import pinecone
from openai import OpenAI
import json

PINECONE_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
INDEX = 'codestyle-semantic-search'

@st.cache_resource
def init_openai():
    client = OpenAI()
    return client
    
@st.cache_resource
def init_key_value():
    with open('./data/mapping.json', 'r') as fp:
        mappings = json.load(fp)
    return mappings

@st.cache_resource
def init_pinecone(index_name):
    # initialize connection to Pinecone vector DB (app.pinecone.io for API key)
    pinecone.init(
        api_key=PINECONE_KEY,
        environment=PINECONE_ENV  # find next to API key in console
    )
    index = pinecone.Index(index_name)
    stats = index.describe_index_stats()
    dims = stats['dimension']
    count = stats['namespaces']['']['vector_count']
    return index, dims, count

def get_embedding(client, text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def create_context(client, mappings, question, index, max_len=3750):
    """
    Find most relevant context for a question via Pinecone search
    """
    q_embed = get_embedding(client, question)
    res = index.query(q_embed, top_k=5, include_metadata=True)
    

    cur_len = 0
    contexts = []
    sources = []

    for row in res['matches']:
        text = mappings[row['id']]
        cur_len += row['metadata']['n_tokens'] + 4
        if cur_len < max_len:
            contexts.append(text)
            sources.append(row['metadata'])
        else:
            cur_len -= row['metadata']['n_tokens'] + 4
            if max_len - cur_len < 200:
                break
    return "\n\n###\n\n".join(contexts), sources

def answer_question(
    client,
    index,
    mappings,
    question="How to name a variable?",
    max_len=3550,
    debug=False,
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context, sources = create_context(
        client,
        mappings,
        question,
        index,
        max_len=max_len,
    )
    if debug:
        print("Context:\n" + context)
        print("\n\n")
    try:
        #print(instruction.format(context, question))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful codestyle assistant. When answering question, use following format: 1. Describe style requirements in one or two sentences, 2. Give a few examples that follow described styleguides, 3. Provide descriptions that you were able to find that clarify why such style was chosen"},
                {"role": "system", "content": f'Codestyle rules: {context}'},
                {"role": "user", "content": question},
            ]
        )
        return response.choices[0].message.content, sources
    except Exception as e:
        print(e)
        return ""

def search(client, index, text_map, query):
    if query != "":
        with st.spinner("Retrieving, please wait..."):
            answer, sources = answer_question(
                client, index, text_map, query,
            )
        # display the answer
        st.write(answer)
        with st.expander("Sources"):
            for source in sources:
                st.write(f"""
                {source['document']} > {source['chapter']} > [{source['section']}]({source['link']})
                """)

st.markdown("""
<link
  rel="stylesheet"
  href="https://fonts.googleapis.com/css?family=Roboto:300,400,500,700&display=swap"
/>
""", unsafe_allow_html=True)

with st.spinner("Connecting to OpenAI..."):
    client = init_openai()

with st.spinner("Connecting to Pinecone..."):
    index, dims, count = init_pinecone(INDEX)
    text_map = init_key_value()

def run():
    st.write("# Codestyle Q&A")
    search = st.container()
    query = search.text_input('Ask a question!', "")
    
    st.sidebar.write(f"""
    ### Info

    **Pinecone index name**: {INDEX}

    **Pinecone index size**: {count}

    **OpenAI embedding model**: *text-search-ada-002*

    **Vector dimensionality**: {dims}

    **OpenAI generation model**: *gpt-3.5-turbo*

    Want to see the original sources that GPT-3.5 is using to generate the answer? No problem, just click on the **Sources** box.
    """)

    if search.button("Go!") or query != "":
        with st.spinner("Retrieving, please wait..."):
            # ask the question
            answer, sources = answer_question(
                client, index, text_map,
                question=query
            )
        # display the answer
        st.write(answer)
        with st.expander("Sources"):
            for source in sources:
                st.write(f"""
                {source['document']} > {source['chapter']} > [{source['section']}]({source['link']})
                """)

if __name__ == "__main__":
    run()

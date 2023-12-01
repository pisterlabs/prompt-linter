# importing  important libraries
import streamlit as st
import cohere
import pinecone
import openai

st.subheader("This is an app to answer questions from the Quran (with references)")

#  storing api keys both cohere and pinecone
pinecone_api_key = st.secrets[pinecone_API_key]
cohere_api_key = st.secrets[cohere_API_key]

# initializing cohere and pinecone
co = cohere.Client(cohere_api_key)

index_name = 'tafsir'
pinecone.init(pinecone_api_key, environment='us-west1-gcp')


# connect to index
index = pinecone.Index(index_name)

# defining the limit of the context
limit = 1600

def retrieve(query):
    xq = co.embed(
        texts=[query],
        model='multilingual-22-12',
        truncate='NONE'
    ).embeddings
    # search pinecone index for context passage with the answer
    xc = index.query(xq, top_k=3, include_metadata=True)
    contexts = [
        x['metadata']['text'] for x in xc['matches']
    ]

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the Query based on the contexts, if it's not in the contexts say 'I don't know the answer'. \n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuery: {query}\nAnswer in the language of Query, if Query is in English Answer in English. Please provide reference Quran verses."
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    return prompt

# st.markdown("### if nothing is written in text box the above mentioned text is by default recommended to you")
    

# st.spinner("Searching for the answer")
# query_with_contexts = retrieve(query)
# query_with_contexts 



# get API key from top-right dropdown on OpenAI website
openai.api_key = st.secrets[openai_API_key]


def complete(prompt):
    # query text-davinci-003
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()

# complete(query_with_contexts)
# st.text(results)

# st.markdown(results)

results = ''

with st.form("my_form"):
    query = st.text_area(":green[Enter Your :question: Question] :point_left:", 
                help= "Ask question on any topic in the Holy Quran",
                height=100,
                placeholder="Ask question on any topic in the Holy Quran")

    submitted = st.form_submit_button("Submit")
    if submitted:
        query_with_contexts = retrieve(query)
        results = complete(query_with_contexts)

st.write(results)

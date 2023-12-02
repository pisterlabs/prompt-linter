import streamlit as st
from streamlit_chat import message
from langchain import OpenAI
#from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain import PromptTemplate
from langchain import LLMChain
import openai
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import pinecone

OPENAI_API_KEY =  st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY =  st.secrets["PINECONE_API_KEY"]
PINECONE_API_ENV =  st.secrets["PINECONE_API_ENV"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]
EMBEDDING_MODEL = 'text-embedding-ada-002'
index_name = PINECONE_INDEX_NAME

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV  # may be different, check at app.pinecone.io
)
# connect to index
index = pinecone.Index(index_name)

@st.cache_resource
def LLM_chain_response():
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template="Answer the question based on the context below. If you cannot answer based on the context, you may use general information about the Lord of the Rings Fellowship of the Ring book or movie. Use Markdown and text formatting to format your answer. \n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:"
    )

    llm = OpenAI(
        temperature=0.3,
        openai_api_key=OPENAI_API_KEY,
        model_name="text-davinci-003",
        max_tokens=128
    )

    chatgpt_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory= ConversationSummaryBufferMemory(llm=llm, max_token_limit=256)
    )
    return chatgpt_chain



# Define the Splade class
class SPLADE:
    def __init__(self, model):
        # check device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForMaskedLM.from_pretrained(model)
        # move to gpu if available
        self.model.to(self.device)

    def __call__(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        inter = torch.log1p(torch.relu(logits[0]))
        token_max = torch.max(inter, dim=0)  # sum over input tokens
        nz_tokens = torch.where(token_max.values > 0)[0]
        nz_weights = token_max.values[nz_tokens]

        order = torch.sort(nz_weights, descending=True)
        nz_weights = nz_weights[order[1]]
        nz_tokens = nz_tokens[order[1]]
        return {
            'indices': nz_tokens.cpu().numpy().tolist(),
            'values': nz_weights.cpu().numpy().tolist()
        }

# Instantiate the Splade class with the path to the model
splade = SPLADE("naver/splade-cocondenser-ensembledistil")

# Define the retrieve function
#@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(3), retry_error_callback=retry_if_not_exception_type(TypeError))

def retrieve(query):
    # retrieve from Pinecone
    res = openai.Embedding.create(input=[query],model=EMBEDDING_MODEL)
    xq = res['data'][0]['embedding']
    sq = splade(query)
    
    # get relevant contexts
    pinecone_res = index.query(xq, top_k=4, include_metadata=True, sparse_vector=sq)
    contexts = [x['metadata']['chunk_text'] for x in pinecone_res['matches']]
    #contexts = [x['metadata']['chunk_text'] for x in pinecone_res['matches'] if x['score'] > 0.8]
    # urls = [x['metadata']['url'] for x in pinecone_res['matches']]
    #print([score['score'] for score in pinecone_res['matches']])

    pinecone_contexts = (
        "\n\n---\n\n".join(contexts)
    )
    return pinecone_contexts

# From here down is all the StreamLit UI.
image = open("./streamlit_app/Pinecone logo white.png", "rb").read()
st.image(image)
st.write("### Lord of the Rings - Fellowship of the Ring Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Why is the ring important in this story?", key="input")
    return input_text

# Main function for the Streamlit app
def main():
    chatgpt_chain = LLM_chain_response()
    user_input = get_text()
    if user_input:
        with st.spinner("Thinking..."):
            query = user_input
            pinecone_contexts = retrieve(query)
            output = chatgpt_chain.predict(input=query + '\nContext: ' + pinecone_contexts)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user", avatar_style="shapes")


if __name__ == "__main__":
    main()

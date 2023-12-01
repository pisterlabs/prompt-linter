import anthropic
import os
import streamlit as st
import tiktoken

from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatAnthropic

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


from langchain.embeddings.openai import OpenAIEmbeddings
from streamlit.logger import get_logger
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.vectorstores import DeepLake
from streamlit_chat import message

st.set_page_config(page_title="Q&A", page_icon="🤖")

st.markdown("# Q&A")

st.markdown(
    """Now that the code has been processed, you can ask it questions.
    
    1. Load the embeddings then chat with Robocop. (eg. dataset name: lido-dao or uniswap-v3)
    2. Click on "Start" to load Robocop and start a conversation.
    3. Type in your question or instruction in the "You:" box and click "Ask" to get an answer.
    """
)

dataset_name = st.text_input(
    label="Dataset name (eg. uniswap-v3)"
)


if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ''

if "activeloop_api_key" not in st.session_state:
    st.session_state["activeloop_api_key"] = ''

if "anthropic_api_key" not in st.session_state:
    st.session_state["anthropic_api_key"] = ''

if "settings_override" not in st.session_state:
    st.session_state["settings_override"] = ''

if "system_message_prompt" not in st.session_state:
    st.session_state["system_message_prompt"] = SystemMessagePromptTemplate.from_template("This is a GitHub repo.")


os.environ['OPENAI_API_KEY'] = st.session_state["openai_api_key"] if st.session_state["settings_override"] else st.secrets.openai_api_key
os.environ['ACTIVELOOP_TOKEN'] = st.session_state["activeloop_api_key"] if st.session_state["settings_override"] else st.secrets.activeloop_api_key
os.environ['ANTHROPIC_API_KEY'] = st.session_state["anthropic_api_key"] if st.session_state["settings_override"] else st.secrets.anthropic_api_key

if os.environ['OPENAI_API_KEY'] == '' or os.environ['ACTIVELOOP_TOKEN'] == '' or os.environ['ANTHROPIC_API_KEY'] == '':
    status = st.info("You have not submitted any API keys yet. Go to the Configure page first.", icon="ℹ️")
else:
    pass

if "generated" not in st.session_state:
    st.session_state["generated"] = ["Hi, I'm Robocop. How may I help you?"]

if "past" not in st.session_state:
    st.session_state["past"] = ["Hi!"]

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [("Hi","Hi, I'm Robocop. How may I help you?")]

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

    
logger = get_logger(__name__)

def count_tokens(question, model):
    count = f'Words: {len(question.split())}'
    if model.startswith("claude"):
        count += f' | Tokens: {anthropic.count_tokens(question)}'
    return count


template = """You are Robocop. Robocop is an expert in identifying security vulnerabilities in smart contracts and blockchain-related codebases. 

Robocop is a technical assistant that provides sophisticated and helpful answer. 

Robocop is trained to analyze all logic with an "attacker" mindset, considering edge cases and extremes. 
It does not focus only on normal use cases.
It reviews code line-by-line in detail, not just at a higher level.
It does not assume any logic is fool proof.

Whenever it finds a vulnerability, Robocop provides a detailed explanation of the vulnerability, a proof of concept of how it might be exploited, and recommended steps to mitigate th risk.

You are auditing a codebase summarized below.
----------------
//REPO_SUMMARY
----------------

Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
{context}
"""

embeddings = 'test'

with st.expander("Advanced settings"):
    distance_metric = st.text_input(
        label="How to measure distance: (L2, L1, max, cos, dot)",
        value="cos"
    )
    model_option = st.selectbox(
        "What model would you like to use?",
        ("claude-v1", 'gpt-3.5-turbo','gpt-4')
    )
    temperature = st.text_input(
        label="Set temperature: 0 (deterministic) to 1 (more random).",
        value="0"
    )
    max_tokens = st.text_input(
        label="Max tokens in the response. (Default: 2,000)",
        value="2000"
    )
    k = st.text_input(
        label="Number of results to return (Default: 10)",
        value="10"
    )
    k_for_mrr = st.text_input(
        label="Number of Documents to fetch to pass to MMR algorithm (Default: 100)",
        value="100"
    )
    maximal_marginal_relevance = st.checkbox(
        label="(Default: True)",
        value=True
    )

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_qa_model(model_option):
    # some notes on memory
    # https://stackoverflow.com/questions/76240871/how-do-i-add-memory-to-retrievalqa-from-chain-type-or-how-do-i-add-a-custom-pr
    dataset_path = f'hub://mrspaghetticode/{dataset_name}'
    
    db = DeepLake(dataset_path=dataset_path, read_only=True, embedding_function=OpenAIEmbeddings(disallowed_special=()))
    retriever = db.as_retriever()

    retriever.search_kwargs['distance_metric'] = distance_metric
    retriever.search_kwargs['k'] = int(k)
    retriever.search_kwargs['maximal_marginal_relevance'] = maximal_marginal_relevance
    retriever.search_kwargs['fetch_k'] = int(k_for_mrr)

    if model_option.startswith("gpt"):
        logger.info('Using OpenAI model %s', model_option)
        qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(
                model_name=model_option, 
                temperature=float(temperature),
                max_tokens=max_tokens
            ),
            retriever=retriever,
            memory=memory,
            verbose=True
        )
    elif os.environ['ANTHROPIC_API_KEY'] != "" and model_option.startswith("claude"):
        logger.info('Using Anthropics model %s', model_option)
        qa = ConversationalRetrievalChain.from_llm(
            ChatAnthropic(
                temperature=float(temperature),
                max_tokens_to_sample=max_tokens
        ),
        retriever=retriever,
        memory=memory,
        verbose=True,
        max_tokens_limit=102400
        )
    return qa

def generate_system_prompt(qa):
    logger.info("Generating System Prompt")
    summary_prompt = f"Provide a short summary (five bullet points max) of the codebase or repository you are auditing {dataset_name}."
    response = qa.run(
        {"question": summary_prompt,
        "chat_history": []
        }
    )
    final_prompt = template.replace("//REPO_SUMMARY", response)
    logger.info(final_prompt)
    return final_prompt
    


def generate_response(prompt, chat_history, qa):
    # maybe use a different chain that includes model retriever, memory)
    # https://python.langchain.com/en/latest/modules/indexes/getting_started.html
    # https://github.com/hwchase17/langchain/discussions/3115
    

    print(qa)
    print("*****")
    print(qa.question_generator.prompt.template)

    qa.question_generator.prompt.template = """
    Given the following conversation and follow up question, rephrase the follow up question to be a standalone question. Ensure that the output is in English.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question: 
    """
    qa.combine_docs_chain.llm_chain.prompt.messages[0] = st.session_state["system_message_prompt"]

    response = qa.run(
        {"question": prompt,
        "chat_history": chat_history
        }
    )
    logger.info('Result: %s', response)
    logger.info(qa.memory)
    return response

def generate_first_response(qa):
    st.session_state["system_message_prompt"] = SystemMessagePromptTemplate.from_template(generate_system_prompt(qa))
    first_prompt = "Please provide an overview of the codebase along with some potential areas to examine for vulnerabilities."
    print(st.session_state["chat_history"])
    first_response = generate_response(first_prompt, st.session_state["chat_history"], qa)
    st.session_state.past.append(first_prompt)
    st.session_state.generated.append(first_response)
    st.session_state.chat_history.append((first_prompt,first_response))

if st.button("🚨 Start 🚨"):
    qa = None
    status = st.info(f'Loading embeddings', icon="ℹ️")
    with st.spinner('Loading Embeddings...'):
        qa = get_qa_model(model_option)
    with st.spinner('Loading Robocop...'):
        status.info(f'Initializing conversation.', icon="ℹ️")
        generate_first_response(qa)
        status.info(f'Ready to chat. Type your question and click on "Ask"', icon="✅")

st.header("Talk to Robocop")

columns = st.columns(3)
with columns[0]:
    button = st.button("Ask")
with columns[1]:
    count_button = st.button("Count Tokens", type='secondary')
with columns[2]:
    clear_history = st.button("Clear History", type='secondary')

if clear_history:
    # Clear memory in Langchain
    memory.clear()
    st.session_state["generated"] = ["Hi, I'm Robocop. How may I help you?"]
    st.session_state["chat_history"] = [("Hi","Hi, I'm Robocop. Ask me anything about the target codebase.")]
    st.session_state["past"] = ["Hi!"]
    st.experimental_rerun()

input_container = st.container()
response_container = st.container()

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text
## Applying the user input box
with input_container:

    user_input = get_text()


## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if button:
        logger.info("Ask button pressed")
        qa = get_qa_model(model_option)
        with st.spinner('Processing...'):
            response = generate_response(user_input, st.session_state["chat_history"], qa)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(response)
            st.session_state.chat_history.append((user_input,response))


        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="initials", seed="jc")
            message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")
    if count_button:
        st.write(count_tokens(user_input, model_option))
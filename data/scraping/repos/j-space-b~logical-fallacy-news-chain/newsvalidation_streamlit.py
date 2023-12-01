# import packages
import warnings
warnings.filterwarnings('ignore')


import sys
sys.path.append('../..')
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, JSONLoader, UnstructuredFileLoader, WebBaseLoader
from langchain.chains import LLMChain, SequentialChain, RetrievalQA
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import streamlit as st, tiktoken

# import 19 logical fallacies
# source of extract: "https://arxiv.org/pdf/2212.07425.pdf
# will be used to evaluate results when using this as a datasource rather than the pdf

st.title("Logical Fallacy Extraction from News Articles")


fallacies = {
    "Logical Fallacy Name": ["Adhominem","Adpopulum","Appeal to Emotion","Fallacy of Extension",
                        "Intentional Fallacy","False Causality","False Dilemma","Hasty Generalization",
                        "Illogical Arrangement","Fallacy of Credibility","Circular Reasoning",
                        "Begging the Question","Trick Question","Overapplying","Equivocation","Amphiboly",
                        "Word Emphasis","Composition","Division"],
    "Description": ["attacks on the character or personal traits of the person making an argument rather than addressing the actual argument and evidence",
                   "the fallacy that something must be true or correct simply because many people believe it or do it, without actual facts or evidence to support",
                   "an attempt to win support for an argument by exploiting or manipulating people's emotions rather than using facts and reason",
                   "making broad, sweeping generalizations and extending the implications of an argument far beyond what the initial premises support",
                   "falsely supporting a conclusion by claiming to understand an author or creator's subconscious intentions without clear evidence",
                   "jumping to conclusions about causation between events or circumstances without adequate evidence to infer a causal relationship",
                   "presenting only two possible options or sides to a situation when there are clearly other alternatives that have not been considered or addressed",
                   "making a broad inference or generalization to situations, people, or circumstances that are not sufficiently similar based on a specific example or limited evidence",
                   "constructing an argument in a flawed, illogical way, so the premises do not connect to or lead to the conclusion properly",
                   "dismissing or attacking the credibility of the person making an argument rather than directly addressing the argument itself",
                 "supporting a premise by simply repeating the premise as the conclusion without giving actual proof or evidence",
                  "restating the conclusion of an argument as a premise without providing actual support for the conclusion in the first place",
                   "asking a question that contains or assumes information that has not been proven or substantiated",
                   "applying a general rule or generalization to a specific case it was not meant to apply to",
                   "using the same word or phrase in two different senses or contexts within an argument",
                   "constructing sentences such that the grammar or structure is ambiguous, leading to multiple interpretations",
                   "shifting the emphasis of a word or phrase to give it a different meaning than intended",
                   "erroneously inferring that something is true of the whole based on the fact that it is true of some part or parts",
                   "erroneously inferring that something is true of the parts based on the fact that it is true of the whole"]
}
json_str = json.dumps(fallacies, indent=4)


with open("fallacies.json", "w") as json_file:
    json_file.write(json_str)


# Replicating above functionality but using Streamlit for input
st.subheader('Enter search terms or URL for analysis:')

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", value="", type="password")
    st.write(
        "Receive an OpenAPI Key here [link](https://platform.openai.com/account/api-keys)")
    serper_api_key = st.text_input("Serper API Key", value="", type="password")
    st.write(
        "Receive an Serper API Key here [link](https://serper.dev/login)")
    num_results = st.number_input("Number of Search Results", min_value=3, max_value=5)
    st.caption("*Search: Analyzes top titles relevant to search terms, URL Lookup: Analyzes specific URL*")

search_query = st.text_input("Search Query", label_visibility="collapsed")
col1, col2 = st.columns(2)

# If the 'Search' button is clicked
if col1.button("Search"):
    # Validate inputs
    if not openai_api_key.strip() or not serper_api_key.strip() or not search_query.strip():
        st.error(f"Please provide the API keys or the missing search terms.")
    else:
        try:
            with st.spinner("Analyzing articles..."):
                # Show the top X relevant news articles from the previous week using Google Serper API
                search = GoogleSerperAPIWrapper(type="news", tbs="qdr:w1", serper_api_key=serper_api_key)
                result_dict = search.results(search_query)
                token_limit = 4000
                if not result_dict['news']:
                    st.error(f"No search results for: {search_query}.")
                else:
                    for i, item in zip(range(num_results), result_dict['news']):
                        url = item.get('link','N/A')
                        if url == 'N/A':
                            continue
                        loader = WebBaseLoader(url) # bs4
                        try:
                            datanews = loader.load()
                            try:
                                # define model
                                llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', max_tokens=1024,
                                                 openai_api_key=openai_api_key)
                                # Chain 1: Summarize any possible logical fallacies in the text of the article

                                template1 = """You are a communications expert who speaks nothing but truth and logic and is \
                                extremely clear for a wide audience.  Given a full news article, your job is to summarize \
                                it accurately and with brevity in one sentence, then find any logical fallacies that \
                                may exist and return examples in no more than 1 sentence per logical fallacy found.  \
                                If more than one logical fallacies are found, return the top 2, in order of logical strength, \
                                unless no logical fallacies are found, in which then state no strong logical fallacies are clearly evident. \
                                Article: {datanews} \
                                Communications expert: 
                                Summary:"""
                                prompt_template1 = PromptTemplate(input_variables=["datanews"], template=template1)
                                chain1 = LLMChain(llm=llm, prompt=prompt_template1, output_key='summary')

                                # Chain 2: Analyze the implications of any logical fallacies in the article in relation to the article summary

                                template2 = """You are an engaging professor who only speaks with truth and sound logic \
                                while clearly conveying a point in as few words as possible.  Given the text of a news article as defined, for the Analysis output:it will be three parts.  The first part is labeled 'Summary' and returns {summary}.  The second part is labeled 'Analysis' and is two sentences.  First, you need \
                                to return the top ranked logical fallacy in the article, among any logical fallacies that may exist, ranked by order of logical strength, \
                                described with brevity in one sentence and confirming this logical fallacy is correct by extracting factual evidence \
                                from the article text, then finally referencing the extracted fact in the description. Be sure to state the strongest logical fallacy might not be strong, \
                                so is only to consider. Create the second sentence of the Analysis output by stating why this fallacy might be dangerous to the public or \
                                especially misleading in the context of the news article, with respect to how other readers could react. \
                                The third part is labeled 'Theoretical Counterfactual', explain any counterfactuals to the summary of the article ({summary}) that could hypothetically be true, \
                                based on logic and the limited facts presented in the article.  If more than one counterfactuals exist, only return  \
                                the top ranked counterfactual, ranked in order of logical strength and feasibility, described with brevity in 1 sentence. \ 
                                Professor: \
                                Summary: {summary}\
                                Analysis: \ 
                                Theoretical Counterfactual: """
                                prompt_template2 = PromptTemplate(input_variables=["summary"], template=template2)
                                chain2 = LLMChain(llm=llm, prompt=prompt_template2, output_key='analysis')

                                overall_chain1 = SequentialChain(chains=[chain1, chain2],
                                    input_variables=["datanews"],
                                    output_variables=["analysis"],
                                    verbose=False)
                                first = (overall_chain1({"datanews":datanews}))
                                second = first['datanews']
                                st.success(f"Critique: \n\nTitle: {item['title']}\nLink: {item['link']}\n\n{first['analysis']}")
                              #  st.success(f"Critique: {item['analysis']}\n\nLink: {item['link']}")
                            except Exception as e:
                                if "token" in str(e).lower():
                                    num_results += 1
                                else:
                                    raise
                        except Exception as e:
                            st.exception(f"Error fetching {item['link']}, exception: {e}")
        except Exception as e:
            st.exception(f"Exception: {e}")

# If 'URL Lookup' button is clicked
if col2.button("URL Lookup"):
    # Validate inputs
    if not openai_api_key.strip() or not serper_api_key.strip() or not search_query.strip():
        st.error(f"Please provide the API keys or missing URL in the search term window.")
    else:
        try:
            with st.spinner("Analyzing articles..."):
                # Show the top X relevant news articles from the URL entered - lookup since URLs change
                search = GoogleSerperAPIWrapper(type="news", tbs="qdr:w1", serper_api_key=serper_api_key)
                # define model
                llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', max_tokens=1024, openai_api_key=openai_api_key)
                result_dict = search.results(search_query)
                token_limit=4000
                if not result_dict['news']:
                    st.error(f"No search results for: {search_query}.")
                else:
                    for i, item in zip(range(num_results), result_dict['news']):
                        url = item.get('link','N/A')
                        if url == 'N/A':
                            continue
                        loader = WebBaseLoader(url) # bs4
                        try:
                            datanews = loader.load()
                            try:

                                # Chain 1: Summarize any possible logical fallacies in the text of the article

                                template1 = """You are a communications expert who speaks nothing but truth and logic and is \
                                extremely clear for a wide audience.  Given a full news article, your job is to summarize \
                                it accurately and with brevity in one sentence, then find any logical fallacies that \
                                may exist and return examples in no more than 1 sentence per logical fallacy found.  \
                                If more than one logical fallacies are found, return the top 2, in order of logical strength, \
                                unless no logical fallacies are found, in which then state no strong logical fallacies are clearly evident. \
                                Article: {datanews} \
                                Communications expert: 
                                Summary:"""
                                prompt_template1 = PromptTemplate(input_variables=["datanews"], template=template1)
                                chain1 = LLMChain(llm=llm, prompt=prompt_template1, output_key='summary')

                                # Chain 2: Analyze the implications of any logical fallacies in the article in relation to the article summary

                                template2 = """You are an engaging professor who only speaks with truth and sound logic \
                                while clearly conveying a point in as few words as possible. Create two output sections: Analysis and Counterfactual. \
                                For the Analysis output, it will be three parts.  The first part is labeled 'Summary' and returns {summary}.  The second part is labeled 'Analysis' and is two sentences.  First, you need \
                                to return the top ranked logical fallacy in the article, among any logical fallacies that may exist, ranked by order of logical strength, \
                                described with brevity in one sentence and confirming this logical fallacy is correct by extracting factual evidence \
                                from the article text, then finally referencing the extracted fact in the description. Be sure to state the strongest logical fallacy might not be strong, \
                                so is only to consider. Create the second sentence of the Analysis output by stating why this fallacy might be dangerous to the public or \
                                especially misleading in the context of the news article, with respect to how other readers could react. \
                                The third part is labeled 'Theoretical Counterfactual', explain any counterfactuals to the summary of the article ({summary}) that could hypothetically be true, \
                                based on logic and the limited facts presented in the article.  If more than one counterfactuals exist, only return  \
                                the top ranked counterfactual, ranked in order of logical strength and feasibility, described with brevity in 1 sentence. \ 
                                Professor: \
                                Summary: {summary}\
                                Analysis: \ 
                                Theoretical Counterfactual: """
                                prompt_template2 = PromptTemplate(input_variables=["summary"], template=template2)
                                chain2 = LLMChain(llm=llm, prompt=prompt_template2, output_key='analysis')

                                overall_chain1 = SequentialChain(chains=[chain1, chain2],
                                                                 input_variables=["datanews"],
                                                                 output_variables=["analysis"],
                                                                 verbose=False)
                                first = (overall_chain1({"datanews": datanews}))
                                second = first['datanews']
                                st.success(
                                    f"Critique: \n\nTitle: {item['title']}\nLink: {item['link']}\n\n{first['analysis']}")
                                    #  st.success(f"Critique: {item['analysis']}\n\nLink: {item['link']}")
                            except Exception as e:
                                if "token" in str(e).lower():
                                    num_results += 1
                                else:
                                    raise
                        except Exception as e:
                            st.exception(f"Error fetching {item['link']}, exception: {e}")
        except Exception as e:
            st.exception(f"Exception: {e}")




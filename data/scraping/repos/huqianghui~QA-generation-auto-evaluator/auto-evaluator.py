import os
import json
import time
from typing import List
import pypdf
import itertools
import text_utils
import pandas as pd
import altair as alt
import streamlit as st
from io import StringIO
import openai
from langchain.llms import Anthropic
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from llama_index import LangchainEmbedding
from langchain.chat_models import AzureChatOpenAI
from langchain.retrievers import SVMRetriever
from langchain.chains import QAGenerationChain
from langchain.retrievers import TFIDFRetriever
from langchain.evaluation.qa import QAEvalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from text_utils import GRADE_DOCS_PROMPT, GRADE_ANSWER_PROMPT, GRADE_DOCS_PROMPT_FAST, GRADE_ANSWER_PROMPT_FAST, GRADE_ANSWER_PROMPT_BIAS_CHECK, GRADE_ANSWER_PROMPT_OPENAI,CHAT_PROMPT
from llama_index import StorageContext, ServiceContext,LLMPredictor,GPTVectorStoreIndex,SimpleDirectoryReader,load_index_from_storage


os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://openaidemo-hu.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_KEY"] = "XXXXXXXXXXX"

OPENAI_API_TYPE = "azure"
OPENAI_API_BASE = "https://openaidemo-hu.openai.azure.com/"
OPENAI_API_VERSION = "2023-03-15-preview"
OPENAI_API_KEY = "XXXXXXXXXXX"
DEPLOYMENT_NAME = "gpt-35-turbo"

openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = "https://openaidemo-hu.openai.azure.com/"
openai.api_key = "XXXXXXXXXXX"


FAISS_FILE_PATH="./faiss_index"
FAISS_INDEX_NAME="finance"

LLAMA_FAISS_FILE_PATH="./llama_src_path"
LLAMA_FAISS_INDEX_FILE_PATH="./llama_faiss_index"
LLAMA_FAISS_INDEX_NAME="finance"

# Keep dataframe in memory to accumulate experimental results
if "existing_df" not in st.session_state:
    summary = pd.DataFrame(columns=['chunk_chars',
                                    'overlap',
                                    'split',
                                    'model',
                                    'retriever',
                                    'embedding',
                                    'num_neighbors',
                                    'Latency',
                                    'Retrieval score',
                                    'Answer score'])
    st.session_state.existing_df = summary
else:
    summary = st.session_state.existing_df

@st.cache_data
def load_docs(files: List) -> str:
    """
    Load docs from files
    @param files: list of files to load
    @return: string of all docs concatenated
    """

    st.info("`读取文档 ...`")
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = pypdf.PdfReader(file_path)
            file_content = ""
            for page in pdf_reader.pages:
                file_content += page.extract_text()
            file_content = text_utils.clean_pdf_text(file_content)
            all_text += file_content
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            file_content = stringio.read()
            all_text += file_content
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_text

@st.cache_data
def generate_eval(text: str, num_questions: int, chunk: int):
    """
    Generate eval set
    @param text: text to generate eval set from
    @param num_questions: number of questions to generate
    @param chunk: chunk size to draw question from in the doc
    @return: eval set as JSON list
    """
    st.info("`生成评估数据集 ...`")
    chunk = round(len(text)/num_questions)
    sub_sequences = split_texts(text,chunk,overlap,split_method)
   
    chain = QAGenerationChain.from_llm(AzureChatOpenAI(openai_api_base=OPENAI_API_BASE,
                                        openai_api_version=OPENAI_API_VERSION,
                                        deployment_name=DEPLOYMENT_NAME,
                                        openai_api_key=OPENAI_API_KEY,
                                        openai_api_type=OPENAI_API_TYPE,
                                        temperature=0),
                                        CHAT_PROMPT)

    eval_set = []
    for i, b in enumerate(sub_sequences):
        try:
            if(i <num_questions):
                qa = chain.run(b)
                eval_set.append(qa)
        except Exception as e:
            st.warning("An exception occurred: %s" % str(e))
            st.warning('Error generating question %s.' % str(i + 1), icon="⚠️")
    eval_set_full = list(itertools.chain.from_iterable(eval_set))
    return eval_set_full

@st.cache_resource
def split_texts(text, chunk_size: int, overlap, split_method: str):
    """
    Split text into chunks
    @param text: text to split
    @param chunk_size:
    @param overlap:
    @param split_method:
    @return: list of str splits
    """
    st.info("`分割文档 ...`")
    if split_method == "RecursiveTextSplitter":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=overlap)
    elif split_method == "CharacterTextSplitter":
        text_splitter = CharacterTextSplitter(separator=" ",
                                              chunk_size=chunk_size,
                                              chunk_overlap=overlap)
    else:
        st.warning("`Split method not recognized. Using RecursiveCharacterTextSplitter`", icon="⚠️")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=overlap)

    split_text = text_splitter.split_text(text)
    return split_text

@st.cache_resource
def make_llm(model_version: str):
    """
    Make LLM from model version
    @param model_version: model_version
    @return: LLN
    """
    if (model_version == "gpt-35-turbo") or (model_version == "gpt-4"):
        chosen_model = AzureChatOpenAI(openai_api_base=OPENAI_API_BASE,
                                        openai_api_version=OPENAI_API_VERSION,
                                        deployment_name=DEPLOYMENT_NAME,
                                        openai_api_key=OPENAI_API_KEY,
                                        openai_api_type=OPENAI_API_TYPE,
                                        temperature=0)
    elif model_version == "anthropic":
        chosen_model = Anthropic(temperature=0)
    else:
        st.warning("`Model version not recognized. Using gpt-35-turbo`", icon="⚠️")
        chosen_model = AzureChatOpenAI(openai_api_base=OPENAI_API_BASE,
                                        openai_api_version=OPENAI_API_VERSION,
                                        deployment_name=DEPLOYMENT_NAME,
                                        openai_api_key=OPENAI_API_KEY,
                                        openai_api_type=OPENAI_API_TYPE,
                                        temperature=0)
    return chosen_model

@st.cache_resource
def make_retriever(splits, retriever_type, embedding_type, num_neighbors, _llm):
    """
    Make document retriever
    @param splits: list of str splits
    @param retriever_type: retriever type
    @param embedding_type: embedding type
    @param num_neighbors: number of neighbors for retrieval
    @param _llm: model
    @return: retriever
    """
    st.info("`创建检索器 ...`")
    # Set embeddings
    if embedding_type == "OpenAI":
        embedding = OpenAIEmbeddings(chunk_size = 1)
    elif embedding_type == "HuggingFace":
        embedding = HuggingFaceEmbeddings()
    else:
        st.warning("`Embedding type not recognized. Using OpenAI`", icon="⚠️")
        embedding = OpenAIEmbeddings(chunk_size = 1)

    # Select retriever
    if retriever_type == "similarity-search":
        try:
            if(os.path.exists(FAISS_FILE_PATH)):
                vector_store = FAISS.load_local(FAISS_FILE_PATH,embedding, FAISS_INDEX_NAME)
            else:
                vector_store = FAISS.from_texts(splits, embedding)
                vector_store.save_local(FAISS_FILE_PATH,FAISS_INDEX_NAME)
        except ValueError:
            st.warning("`Error using OpenAI embeddings (disallowed TikToken token in the text). Using HuggingFace.`",
                       icon="⚠️")
            vector_store = FAISS.from_texts(splits, HuggingFaceEmbeddings())
        retriever_obj = vector_store.as_retriever(k=num_neighbors)
    elif retriever_type == "SVM":
        retriever_obj = SVMRetriever.from_texts(splits, embedding)
    elif retriever_type == "TF-IDF":
        retriever_obj = TFIDFRetriever.from_texts(splits)
    elif retriever_type == "Llama-Index":
        
        llm_predictor = LLMPredictor(llm=llm)
        langchainEmbedding = LangchainEmbedding(OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1, openai_api_version="2023-03-15-preview"))
        # langchainEmbedding = LangchainEmbedding(OpenAIEmbeddings(
        #     model="text-embedding-ada-002",
        #     deployment="text-embedding-ada-002",
        #     openai_api_base=OPENAI_API_BASE,
        #     openai_api_type=OPENAI_API_TYPE, 
        #     chunk_size=1))
        
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=langchainEmbedding)
        if(os.path.exists(LLAMA_FAISS_FILE_PATH)):
            # rebuild storage context
            documents = SimpleDirectoryReader(LLAMA_FAISS_FILE_PATH).load_data()
            # load index
            index = GPTVectorStoreIndex.from_documents(documents,service_context=service_context)
            retriever_obj = index.as_query_engine(service_context=service_context, verbose=True)
            # storage_context = StorageContext.from_defaults(persist_dir=LLAMA_FAISS_INDEX_FILE_PATH)
            # index = load_index_from_storage(storage_context)
            # retriever_obj = index.as_query_engine()
        else:
            os.makedirs(LLAMA_FAISS_FILE_PATH)
            for i, split in enumerate(splits):
                with open(f"{LLAMA_FAISS_FILE_PATH}/{i}.txt", 'w') as fp:
                    fp.write(split)

            # Load all  documents
            splits_docs = SimpleDirectoryReader(LLAMA_FAISS_FILE_PATH).load_data()
            index = GPTVectorStoreIndex.from_documents(splits_docs,service_context=service_context,verbose=True)
            retriever_obj = index.as_query_engine(service_context=service_context, verbose=True)
            os.makedirs(LLAMA_FAISS_INDEX_FILE_PATH)
            index.storage_context.persist(persist_dir = LLAMA_FAISS_INDEX_FILE_PATH)
    else:
        st.warning("`Retriever type not recognized. Using SVM`", icon="⚠️")
        retriever_obj = SVMRetriever.from_texts(splits, embedding)
    return retriever_obj

def make_chain(llm, retriever, retriever_type: str) -> RetrievalQA:
    """
    Make chain
    @param llm: model
    @param retriever: retriever
    @param retriever_type: retriever type
    @return: chain (or return retriever for Llama-Index)
    """
    st.info("`创建langchain ...`")
    if retriever_type == "Llama-Index":
        qa = retriever
    else:
        qa = RetrievalQA.from_chain_type(llm,
                                         chain_type="stuff",
                                         retriever=retriever,
                                         input_key="question")
    return qa

def grade_model_answer(predicted_dataset: List, predictions: List, grade_answer_prompt: str) -> List:
    """
    Grades the distilled answer based on ground truth and model predictions.
    @param predicted_dataset: A list of dictionaries containing ground truth questions and answers.
    @param predictions: A list of dictionaries containing model predictions for the questions.
    @param grade_answer_prompt: The prompt level for the grading. Either "Fast" or "Full".
    @return: A list of scores for the distilled answers.
    """
    # Grade the distilled answer
    st.info("`模型答案评分 ...`")
    # Set the grading prompt based on the grade_answer_prompt parameter
    if grade_answer_prompt == "Fast":
        prompt = GRADE_ANSWER_PROMPT_FAST
    elif grade_answer_prompt == "Descriptive w/ bias check":
        prompt = GRADE_ANSWER_PROMPT_BIAS_CHECK
    elif grade_answer_prompt == "OpenAI grading prompt":
        prompt = GRADE_ANSWER_PROMPT_OPENAI
    else:
        prompt = GRADE_ANSWER_PROMPT

    # Create an evaluation chain
    eval_chain = QAEvalChain.from_llm(
        llm=AzureChatOpenAI(openai_api_base=OPENAI_API_BASE,
                                        openai_api_version=OPENAI_API_VERSION,
                                        deployment_name=DEPLOYMENT_NAME,
                                        openai_api_key=OPENAI_API_KEY,
                                        openai_api_type=OPENAI_API_TYPE,
                                        temperature=0),
        prompt=prompt
    )

    # Evaluate the predictions and ground truth using the evaluation chain
    graded_outputs = eval_chain.evaluate(
        predicted_dataset,
        predictions,
        question_key="question",
        prediction_key="result"
    )

    return graded_outputs

def grade_model_retrieval(gt_dataset: List, predictions: List, grade_docs_prompt: str):
    """
    Grades the relevance of retrieved documents based on ground truth and model predictions.
    @param gt_dataset: list of dictionaries containing ground truth questions and answers.
    @param predictions: list of dictionaries containing model predictions for the questions
    @param grade_docs_prompt: prompt level for the grading. Either "Fast" or "Full"
    @return: list of scores for the retrieved documents.
    """
    # Grade the docs retrieval
    st.info("`检索到的文档进行相关性评分...`")

    # Set the grading prompt based on the grade_docs_prompt parameter
    prompt = GRADE_DOCS_PROMPT_FAST if grade_docs_prompt == "Fast" else GRADE_DOCS_PROMPT

    # Create an evaluation chain
    eval_chain = QAEvalChain.from_llm(
        llm=AzureChatOpenAI(openai_api_base=OPENAI_API_BASE,
                                        openai_api_version=OPENAI_API_VERSION,
                                        deployment_name=DEPLOYMENT_NAME,
                                        openai_api_key=OPENAI_API_KEY,
                                        openai_api_type=OPENAI_API_TYPE,
                                        temperature=0),
        prompt=prompt
    )

    # Evaluate the predictions and ground truth using the evaluation chain
    graded_outputs = eval_chain.evaluate(
        gt_dataset,
        predictions,
        question_key="question",
        prediction_key="result"
    )
    return graded_outputs

def run_evaluation(chain, retriever, eval_set, grade_prompt, retriever_type, num_neighbors):
    """
    Runs evaluation on a model's performance on a given evaluation dataset.
    @param chain: Model chain used for answering questions
    @param retriever:  Document retriever used for retrieving relevant documents
    @param eval_set: List of dictionaries containing questions and corresponding ground truth answers
    @param grade_prompt: String prompt used for grading model's performance
    @param retriever_type: String specifying the type of retriever used
    @param num_neighbors: Number of neighbors to retrieve using the retriever
    @return: A tuple of four items:
    - answers_grade: A dictionary containing scores for the model's answers.
    - retrieval_grade: A dictionary containing scores for the model's document retrieval.
    - latencies_list: A list of latencies in seconds for each question answered.
    - predictions_list: A list of dictionaries containing the model's predicted answers and relevant documents for each question.
    """
    st.info("`运行评估 ...`")
    predictions_list = []
    retrieved_docs = []
    gt_dataset = []
    latencies_list = []

    for data in eval_set:

        # Get answer and log latency
        start_time = time.time()
        if retriever_type != "Llama-Index":
            predictions_list.append(chain(data))
        elif retriever_type == "Llama-Index":
            answer = chain.query(data["question"])
            predictions_list.append({"question": data["question"], "answer": data["answer"], "result": answer.response})
        gt_dataset.append(data)
        end_time = time.time()
        elapsed_time = end_time - start_time
        latencies_list.append(elapsed_time)

        # Retrieve docs
        retrieved_doc_text = ""
        if retriever_type == "Llama-Index":
            for i, doc in enumerate(answer.source_nodes):
                retrieved_doc_text += "Doc %s: " % str(i + 1) + doc.node.text + " "

        else:
            docs = retriever.get_relevant_documents(data["question"])
            for i, doc in enumerate(docs):
                retrieved_doc_text += "Doc %s: " % str(i + 1) + doc.page_content + " "

        retrieved = {"question": data["question"], "answer": data["answer"], "result": retrieved_doc_text}
        retrieved_docs.append(retrieved)

    # Grade
    answers_grade = grade_model_answer(gt_dataset, predictions_list, grade_prompt)
    retrieval_grade = grade_model_retrieval(gt_dataset, retrieved_docs, grade_prompt)
    return answers_grade, retrieval_grade, latencies_list, predictions_list

st.set_page_config(page_title="QA生成&效果评估", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)
# Auth
st.sidebar.image("img/diagnostic.jpg")

with st.sidebar.form("user_input"):
    num_eval_questions = st.select_slider("`Number of eval questions`",
                                          options=[1,2,3,4,5,6,7,8,9,10], value=3)

    chunk_chars = st.select_slider("`Choose chunk size for splitting`",
                                   options=[500, 750, 1000, 1500, 2000], value=1000)

    overlap = st.select_slider("`Choose overlap for splitting`",
                               options=[0, 50, 100, 150, 200], value=100)

    split_method = st.radio("`Split method`",
                            ("RecursiveTextSplitter",
                             "CharacterTextSplitter"),
                            index=0)

    model = st.radio("`Choose model`",
                     ("gpt-35-turbo",
                      "gpt-4",
                      "anthropic"),
                     index=0)

    retriever_type = st.radio("`Choose retriever`",
                              ("TF-IDF",
                               "SVM",
                               "Llama-Index",
                               "similarity-search"),
                              index=3)

    num_neighbors = st.select_slider("`Choose # chunks to retrieve`",
                                     options=[3, 4, 5, 6, 7, 8])

    embeddings = st.radio("`Choose embeddings`",
                          ("HuggingFace",
                           "OpenAI"),
                          index=1)

    grade_prompt = st.radio("`Grading style prompt`",
                            ("Fast",
                             "Descriptive",
                             "Descriptive w/ bias check",
                             "OpenAI grading prompt"),
                            index=0)

    submitted = st.form_submit_button("Submit evaluation")

# App
st.header("`QA生成 & 效果评估`")
st.info(
    "`这是一个用于问答评估的工具。对于给定文档，将自动生成一个问题-答案，并且对正确率，文档相关性，延迟等进行评估。 "
    "使用您选择的langchain参数设置进行评估，并记录使用不同配置的实验。 "
    "（可选择项：您可以提供自己的评估集（以JSON格式）。请参阅docs/karpathy-pod-eval.json获取示例).`")

with st.form(key='file_inputs'):
    uploaded_file = st.file_uploader("`请上传要评估的文件（.txt 或 .pdf 格式）:` ",
                                     type=['pdf', 'txt'],
                                     accept_multiple_files=True)

    uploaded_eval_set = st.file_uploader("`[可选] 请上传评估集文件（.json 格式）:` ",
                                         type=['json'],
                                         accept_multiple_files=False)

    submitted = st.form_submit_button("Submit files")

if uploaded_file:

    # Load docs
    text = load_docs(uploaded_file)
    # Generate num_eval_questions questions, each from context of 3k chars randomly selected
    if not uploaded_eval_set:
        eval_set = generate_eval(text, num_eval_questions, 3000)
    else:
        eval_set = json.loads(uploaded_eval_set.read())
    # Split text
    splits = split_texts(text, chunk_chars, overlap, split_method)
    # Make LLM
    llm = make_llm(model)
    # Make vector DB
    retriever = make_retriever(splits, retriever_type, embeddings, num_neighbors, llm)
    # Make chain
    qa_chain = make_chain(llm, retriever, retriever_type)
    # Grade model
    graded_answers, graded_retrieval, latency, predictions = run_evaluation(qa_chain, retriever, eval_set, grade_prompt,
                                                                      retriever_type, num_neighbors)

    # Assemble outputs
    d = pd.DataFrame(predictions)
    d['answer score'] = [g['text'] for g in graded_answers]
    d['docs score'] = [g['text'] for g in graded_retrieval]
    d['latency'] = latency

    # Summary statistics
    mean_latency = d['latency'].mean()
    correct_answer_count = len([text for text in d['answer score'] if "INCORRECT" not in text])
    correct_docs_count = len([text for text in d['docs score'] if "Context is relevant: True" in text])
    percentage_answer = (correct_answer_count / len(graded_answers)) * 100
    percentage_docs = (correct_docs_count / len(graded_retrieval)) * 100

    st.subheader("`运行结果`")
    st.info(
        "`程序将根据以下标准对链路进行评分：1/ 相对于问题的检索到的文档的相关性，以及 2/ 相对于真实答案，对总结后的答案进行评分。"
        " 您可以在text_utils中查看（并更改）用于评分的提示信息`")
    st.dataframe(data=d, use_container_width=True)

    # Accumulate results
    st.subheader("`结果聚合`")
    st.info(
        "`检索和答案分数是由LLM评分器认定为相关的检索文档的百分比 (relative to the question) "
        "以及相对于真实答案认定为相关的总结答案的百分比 (relative to ground truth answer)"
        "点的大小对应于检索和答案总结的延迟时间（以秒为单位） (larger circle = slower)`")
    new_row = pd.DataFrame({'chunk_chars': [chunk_chars],
                            'overlap': [overlap],
                            'split': [split_method],
                            'model': [model],
                            'retriever': [retriever_type],
                            'embedding': [embeddings],
                            'num_neighbors': [num_neighbors],
                            'Latency': [mean_latency],
                            'Retrieval score': [percentage_docs],
                            'Answer score': [percentage_answer]})
    summary = pd.concat([summary, new_row], ignore_index=True)
    st.dataframe(data=summary, use_container_width=True)
    st.session_state.existing_df = summary

    # Dataframe for visualization
    show = summary.reset_index().copy()
    show.columns = ['expt number', 'chunk_chars', 'overlap',
                    'split', 'model', 'retriever', 'embedding', 'num_neighbors', 'Latency', 'Retrieval score',
                    'Answer score']
    show['expt number'] = show['expt number'].apply(lambda x: "Expt #: " + str(x + 1))
    c = alt.Chart(show).mark_circle().encode(x='Retrieval score',
                                             y='Answer score',
                                             size=alt.Size('Latency'),
                                             color='expt number',
                                             tooltip=['expt number', 'Retrieval score', 'Latency', 'Answer score'])
    st.altair_chart(c, use_container_width=True, theme="streamlit")

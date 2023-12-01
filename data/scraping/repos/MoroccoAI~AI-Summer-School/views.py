from django.shortcuts import render
from django.http import HttpResponseRedirect, JsonResponse
from django.urls import reverse


import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.document_loaders.pdf import PDFPlumberLoader
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import pypdf
 
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain
import wikipediaapi
from langchain.memory import ConversationBufferMemory 

#AI Chatbot stuff go here 







# Create your views here.

SYLLABUS = [{'name': "1 Introduction ", 'completed': False}, 
            {'name': "2 Arrays, Iteration, Invariants", 'completed': False},
            {'name': "3 Lists, Recursion, Stacks, Queues", 'completed': False},
            {'name': "4 Searching", 'completed': False},
            {'name': "5 Efficiency and Complexity", 'completed': False},
            {'name': "6 Trees, Queues", 'completed': False},
            {'name': "7 Binary Search Trees", 'completed': False},
            {'name': "8 Priority Queues and Heap Trees, Queues", 'completed': False},
            {'name': "9 Sorting", 'completed': False},
            {'name': "10 Hash Tables", 'completed': False},
            {'name': "11 Graphs", 'completed': False}
            ]

QUESTION_ANSWER = {"question": "What is a for loop used for", "answer": {"1": "To infinitely loop", "2": "To loop for a determined number of times",
                                                                           "3": "There is no such thing as a for loop", "4":"tHE HELL is that?"}}






device = "cuda" if torch.cuda.is_available() else "cpu" 
pdf_path = "chatbot/dsa-1.pdf" 

EMB_SBERT_MPNET_BASE = "sentence-transformers/all-mpnet-base-v2"
huggingfacehub_api_token = "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"
LLM_FLAN_T5_LARGE = "google/flan-t5-large"
config = {"persist_directory":"store",
        "load_in_4bit":True,
        "load_in_8bit":True,
        "embedding" : EMB_SBERT_MPNET_BASE,
        "llm":LLM_FLAN_T5_LARGE,
}



def create_sbert_mpnet():
        return HuggingFaceEmbeddings(model_name=EMB_SBERT_MPNET_BASE, model_kwargs={"device": device})


#Defining a default prompt for flan models
def set_pdfqa_prompt_template_and_memory(qa):
    custum_pdf_prompt = """ Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum, keep the answer as concise as possible and  don't repeate the sentences.
    Always say "thanks for asking!" at the end of the answer.
    context: {context}
    chat history : {chat_history}  
    Question: {question}
    Answer:"""

    qa.combine_docs_chain.llm_chain.prompt = PromptTemplate(
        template=custum_pdf_prompt, input_variables=["context", "question","chat_history"]
    )


def search(topic):
    # Wikipedia API setup


    wiki_html = wikipediaapi.Wikipedia(
        user_agent='MyProjectName (merlin@example.com)',
            language='en',
            extract_format=wikipediaapi.ExtractFormat.HTML
    )
    page_py = wiki_html.page(topic)
    if page_py.exists():
        title = page_py.title
        # text = page_py.text
        text = page_py.fullurl
    else:
        title = "Not Found"
        text = "Please formulate your question"
    return (title, text)

def hamming_distance(seq1, seq2):
    min_ = min(len(seq1),len(seq2))
    seq1 =seq1[:min_]
    seq2 =seq2[:min_]

    distance = sum(c1 != c2 for c1, c2 in zip(seq1, seq2))
    similarity = 1 - (distance / len(seq1))
    return  abs(similarity)

def get_keywords(sentence):
    # Tokenize the sentence
    tokens = word_tokenize(sentence.lower())

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

    # Part-of-speech tagging
    tagged_tokens = pos_tag(filtered_tokens)

    # Select keywords based on POS tags (Nouns, Adjectives, and Verbs)
    keywords = [word for word, tag in tagged_tokens if tag.startswith('N') or tag.startswith('J') or tag.startswith('V')]

    return keywords

def get_answer(question,qa):
    qa.combine_docs_chain.verbose = False
    r= qa({"question":question})
    similarty1 = hamming_distance(r['answer'].lower(), qa.retriever.vectorstore.similarity_search(question)[-1].page_content.lower())
    similarty2 = hamming_distance(r['answer'].lower(), question.lower())
    if  similarty1 > 0.001:
        if similarty2 > 0.01:
            r =r['answer'] 
        else:
            l = search(' '.join(get_keywords(question)))[-1]
            r = "To see more click the link: <a hreaf='"+l+"' >"+l+"</a>"
    else:
            l = search(' '.join(get_keywords(question)))[-1]
            r = "To see more click the link: <a hreaf='"+l+"' >"+l+"</a>"
        
    # print("similarty", similarty1, similarty2, "key words", get_keywords(question))
    return r


def create_flan_t5_large(load_in_nbit=False):
            # Wrap it in HF pipeline for use with LangChain
            model=config["llm"]

            tokenizer = AutoTokenizer.from_pretrained(model)
            return pipeline(
                task="text2text-generation",
                model=model,
                tokenizer = tokenizer,
                # max_new_tokens=150,
                model_kwargs={"device_map": device  
                # , "load_in_4bit": load_in_nbit
                            ,"max_length": 1024
                            ,"temperature": 0.0}
            )


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,input_key="question")

def get_retrievalQa(context,embedding, pipe):
    # loader = PDFPlumberLoader(pdf_path)
    # documents = loader.load()

    # Split documents and create text snippets 
    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    texts = text_splitter.split_documents([context])
    text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20, encoding_name="cl100k_base")  # This the encoding for text-embedding-ada-002
    texts = text_splitter.split_documents(texts)

    persist_directory = config["persist_directory"]
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
    retriever = vectordb.as_retriever(search_kwargs={"k":3})
    
    return ConversationalRetrievalChain.from_llm(llm= HuggingFacePipeline(pipeline=pipe), retriever=retriever,
    get_chat_history=lambda h : h, memory=memory)



####Website Stuff

def index(request):
    return render(request, 'chatbot/index.html', {
            "syllabus": SYLLABUS,
            "qa": QUESTION_ANSWER,
            "text":  x[current_stat]
    })



global pipe 

def initialize(request):

    load_in_4bit = config["load_in_4bit"]
    load_in_8bit = config["load_in_8bit"]
    pipe = create_flan_t5_large(load_in_nbit=load_in_4bit)

    return HttpResponseRedirect(reverse('index'))  





loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()
splitter = RecursiveCharacterTextSplitter(chunk_size = 5000,
    chunk_overlap  = 100,
    length_function = len)
x= splitter.split_documents(pages)
embedding = create_sbert_mpnet()

load_in_4bit = config["load_in_4bit"]
load_in_8bit = config["load_in_8bit"]
pipe = create_flan_t5_large(load_in_nbit=load_in_4bit) 
current_stat = 0
end_of_course = False


def update_state():
    if current_stat<len(x):
        current_stat+=1
    else:
        end_of_course = True



def get_text(request, message):
    
    current_lesson = x[current_stat]
    qa = get_retrievalQa(current_lesson,embedding, pipe)
    set_pdfqa_prompt_template_and_memory(qa)

    q = message
    r = get_answer(q,qa)

    text = "This is the text sent from the Django API!"
    return JsonResponse({'text': r, "message": message}, safe=False)             
  

             
def get_materials():
    # from langchain.chains import LLMChain

    # prompt = PromptTemplate(
    # input_variables=["context"],
    # template="""Use the following piece of context to extrcat the the main chapters and its subsections at the end.
    # Don't try to make up an answer.
    # Use one sentences for each chapter and subsection , keep the answer as concise as possible and  don't repeate the sentences.
    # context: {context}
    # chapters:
    # """,
    # )


    
    # chain = LLMChain(llm=pipe, prompt=prompt)

    # # Run the chain only specifying the input variable.
    # print(chain.run("colorful socks")
    pass
    # return JsonResponse({"materials": x})





from dotenv import load_dotenv
import os
import re
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import  FAISS
from langchain.chains.question_answering import load_qa_chain

# Load environment variables from .env file
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

import openai

openai.api_key = api_key   

def get_method_title(title_list):
    sample_title_list = """['Visual Chain-of-Thought Diffusion Models',
 '1. Introduction',
 '2. Background',
 '3. Conditional vs. unconditional DGMs',
 '4. Method',
 '5. Experiments',
 '6. Related work',
 '7. Discussion and conclusion',
 '']"""
    
    sample_title_list2 ="""
['Chain of Thought Prompt Tuning for Vision-Language Models',
 '1. Introduction',
 '2. Related Works',
 '2.1. Vision Language Models',
 '2.2. Prompt Tuning in Vision-Language Models',
 '2.3. Chain of Thought',
 '3. Method',
 '3.1. Problem Formulation',
 '3.2. Method Overview',
 '3.3. Chain of Thought: Prompt Chaining',
 '3.4. Self Adaptive Chain Controller',
 '3.5. Meta-Nets Chaining',
 '4. Experiments',
 '4.1. Tasks',
 '4.2. Experimental Settings',
 '4.3. Implementation Details',
 '4.4. Results and Analysis',
 '4.5. Ablation Study',
 '5. Conclusion and Limitation',
 '']"""
    chat_history = [
        {'role': 'system', 'content': 'you are a helpful assistant.'},
        {'role': 'user', 'content': "given a list(index starts from 0) of titles from a research paper, between which 2 FIRST-LEVEL titles the content is most likely talking about methodology? ANSWER IN FORMAT 'the INDEX of the start title in list:the INDEX of the end title in list'" + '"""' + sample_title_list + '"""'},
        {'role': 'assistant', 'content': "4(4. Method):5(5. Experiments)"},
        {'role': 'user', 'content':  sample_title_list2 },
        {'role': 'assistant', 'content': "6(3. Method):12(4. Experiments)"},
        {'role': 'user', 'content': '"""' + str(title_list) + '"""'},
        {'role': 'assistant', 'content': ""},
    ]

    response = openai.ChatCompletion.create(
        #model="gpt-3.5-turbo",
        model = "gpt-4",
        messages=chat_history
    )

    answer = response.choices[0].message.content
    return answer

#--------------------------------------------------------------------------#
def get_answer(docsearch, query):
    docs = docsearch.similarity_search(query, top_k=3)
    # # convert it into a string
    # docs_str = docs.__str__()
        
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    chain = load_qa_chain(llm=llm, chain_type= "stuff")
    answer = chain.run(input_documents = docs, question = query)
    return answer
    # chat_history = [
    #     {'role': 'system', 'content': 'you are a helpful assistant.'},
    #     {'role': 'user', 'content': "Answer the query from provided texts under a condition. the condition is if the texts contains words like 'Fig','Figure', 'Table', keep these words in final answer" + 'Provided texts: """' + docs_str + '"""' + ' Query: """' + query + '"""'},
    #     {'role': 'assistant', 'content': ""},
    # ]

    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     #model="gpt-4",
    #     messages=chat_history
    # )

    # answer = response.choices[0].message.content
    # return answer

def ask_question(QA_pair):
    filtered_QA_pair = {}
    #replace the 4th question with the method summary
    #QA_pair["Methodology"] = method_summary

    # Iterate over the original dictionary and filter out pairs with answers containing fewer than 25 words
    for q, a in QA_pair.items():
        if len(a.split()) >= 25:
            filtered_QA_pair[q] = a

    chat_history = [
        {'role': 'system', 'content': 'you are a helpful assistant.'},
        {'role': 'user', 'content': "given a dictionary of question-answer pairs, please distill the useful information and generate ONE new concise and structured paragraph under the subtitle start with ### Overview:\n" + '"""' + str(filtered_QA_pair) + '"""'},
        {'role': 'assistant', 'content': ""},
    ]

    response = openai.ChatCompletion.create(
        #model="gpt-3.5-turbo",
        model="gpt-4",
        messages=chat_history
    )

    answer = response.choices[0].message.content
    return answer, filtered_QA_pair


def generate_summary(raw_text, api_key):
    QA_pair = {}

    if "References" in raw_text:
        raw_text = re.sub(r'References.*$', '', raw_text)


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # download embeddings from OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    docsearch = FAISS.from_texts(texts, embeddings)

    # read the questions from unconditional_Q.txt and store them in a list
    with open('unconditional_Q.txt', 'r') as f:
        questions = f.readlines()
    query_list = [q.strip() for q in questions]

    answer_list = []
    for query in query_list:
        ans = get_answer(docsearch, query)
        answer_list.append(ans)

    # simplified_query_list
    simplified_query_list = ['Research Problem','Motivation','Contribution','Datasets', 'Experiments','Ablation Study', 'Conclusion']
    for i in range(len(simplified_query_list)):
        QA_pair[simplified_query_list[i]] = answer_list[i]

    # # join the answer together
    # whole_answer = ' '.join(answer_list)
    print("get all the QA pairs")
    

    openai.api_key = api_key
    try:
        answer0, filtered_QA_pair = ask_question(QA_pair)
    except Exception as e:
        print(f"An error occurred while communicating with OpenAI: {e}")
    print("get the initial summary")
    return docsearch, answer0, filtered_QA_pair

#-------------------------------------------------------------------------
def get_method_summary(method_text, method_key):
    # when method_key is not none
    if method_key :
        prompt = "This is the text extracted from a PDF, Now generate a summary of this part, use plain English to explain the idea clearly. NOTE that there are some references to"  +  str(method_key) + ". when generate summary, try to KEEP ALL those references."
        #prompt = "This is the methodology paragraph extracted from a PDF, use plain English to explain the idea clearly. \nNOTE: There are some figure/table references to " +  str(method_key) + ". When generate summary, please KEEP ALL in-text references in their original location."
        #prompt = 'This is the methodology paragraph extracted from a PDF, use plain English to generate a clear summary that includes all references to charts or tables. Ensure that the summary retains the details mentioned in any figures, such as "See Figure 1" or "Refer to Table 2." Capture the essence of the information, and do not omit any details related to visual elements. Maintain coherence and conciseness in the generated summary.'
    else:
        prompt = "This is the methodology paragraph extracted from a PDF, use plain English to explain the idea clearly."
    chat_history = [
        {'role': 'system', 'content': 'you are a helpful assistant.'},
        #{'role': 'user', 'content': '"""' + str(method_text) + '"""' + prompt},
        {'role': 'user', 'content': '"""' + str(method_text) + '"""' + prompt },
        {'role': 'assistant', 'content': ""},
    ]

    response = openai.ChatCompletion.create(
        #model="gpt-3.5-turbo",
        model="gpt-4",
        messages=chat_history
    )

    answer = response.choices[0].message.content
    return answer

#-------------------------------------------------------------------------
def split_text(method_text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    text_list = text_splitter.split_text(method_text)

    return text_list

#-------------------------------------------------------------------------
# ## to tell if a pic should be added to the summary based on the caption[GPT]
# def should_add_pic(caption):
#     prompt = """
#     here is the classification criteria:
#     1. class A:
#    - *Model Structure*: Graphs that depict the architecture, framework, or structure of the model, system, or algorithm discussed in the paper. These graphs are vital for understanding the core of the research.
#    - *Methodology*: Graphs that illustrate the methods, procedures, or experimental setups used in the study. These can include flowcharts, process diagrams, or diagrams explaining data collection.
#    - *Key Results*: Graphs that present significant, key results, such as groundbreaking findings, major trends, or essential statistical insights. These are the most crucial findings in the paper.

#     2. class B: 
#    - *Supporting Details*: Graphs that provide supplementary or supporting information without directly conveying the core model, methodology, or key findings. They might include additional data, reference tables, or less significant visual aids.
#    - *Contextual Illustrations*: Graphs that serve to provide context or illustrate general concepts but do not contain critical information for summarizing the paper's main points.

#     Based on following graph caption, which class is more possible? keep answer short like just 'A' or 'B':
# """

#     chat_history = [
#         {'role': 'system', 'content': 'you are a helpful assistant.'},
#         {'role': 'user', 'content': prompt + '"""' + caption + '"""'},
#         {'role': 'assistant', 'content': "The class is:"},
#     ]

#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         #model="gpt-4",
#         messages=chat_history
#     )

#     answer = response.choices[0].message.content

#     return answer
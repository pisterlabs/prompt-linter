import pandas as pd
from sklearn.metrics import f1_score
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import warnings
warnings.filterwarnings("ignore")

DB_FAISS_PATH = 'vectorstore/db_faiss'

## Default LLaMA-2 prompt style
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

sys_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. """

instruction = """CONTEXT:/n/n {context}/n

Question: {question}"""
get_prompt(instruction, sys_prompt)

prompt_template = get_prompt(instruction, sys_prompt)

llama_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": llama_prompt}

# custom_prompt_template = """ Retrieve information only from the following documents: DFPDS2021, DPM2009, and GFR2017, which are stored in 'vectorstore/db_faiss'.

# Context: {context}
# Question: {question}

# Please ensure that your answer is based solely on the information available in these documents. If you don't know the answer, please respond with 'I don't know.'
# """


# def set_custom_prompt():
#     """
#     Prompt template for QA retrieval for each vectorstore
#     """
#     prompt = PromptTemplate(template=custom_prompt_template,
#                             input_variables=['context', 'question'])
#     return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold":0.3,"k": 4}),
                                       return_source_documents=True,
                                       chain_type_kwargs=chain_type_kwargs
                                       )
    return qa_chain

# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q4_K_S.bin",
        model_type="llama",
        max_new_tokens=1024,
        max_tokens=1024,
        repetition_penalty= 1.1,
        temperature=0.5,
        # top_k=50,
        # top_p=0.9
    )
    return llm

# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = llama_prompt
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Load the custom Q&A dataset
custom_dataset = pd.read_csv('Q&A.csv')  # Load your Q&A dataset here

def evaluate_f1_score():
    predicted_answers = []
    actual_answers = []
    questions = []

    # Iterate through the custom dataset
    for question in custom_dataset['Question']:
        # Get the response from the QA system (you should modify this part)
        qa_result = qa_bot()  # Call your QA system here
        response = qa_result({'query': question})['result']  # Extract the 'result' field

        # Append the predicted answer from the system to the list
        predicted_answers.append(response)

        # Collect the actual answers from the dataset
        actual_answer = custom_dataset.loc[custom_dataset['Question'] == question, 'Answer'].values[0]
        actual_answers.append(actual_answer)

        # Collect the questions for the DataFrame
        questions.append(question)

    # Create a DataFrame
    evaluation_df = pd.DataFrame({'Question': questions, 'Actual Answer': actual_answers, 'Predicted Answer': predicted_answers})

    # Calculate F1 score
    f1 = f1_score(actual_answers, predicted_answers, average='micro')

    return evaluation_df, f1

# Evaluate the system and get the DataFrame and F1 score
evaluation_df, f1_score_result = evaluate_f1_score()

# Print the DataFrame
print("Evaluation DataFrame:")
print(evaluation_df)

# Print the F1 score
print(f"F1 Score: {f1_score_result}")

# Save the DataFrame to a CSV file
evaluation_df.to_csv('evaluation_results2.csv', index=False)  # Save the DataFrame to a CSV file named 'evaluation_results.csv'
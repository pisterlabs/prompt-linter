import os
from flask import Flask, request, jsonify
import textract
from tqdm import tqdm

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate


app = Flask(__name__)
os.environ['OPENAI_API_KEY'] = 'YOUR OPEN AI API KEY'


@app.route('/upload', methods = ['POST','GET'])
def uploadFiles():
    json_ = request.json
    folder_path = json_['FolderPath']
    save_path = json_['SavePath']

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    try:
        files = os.listdir(folder_path)

        for file in tqdm(files):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                doc = textract.process(file_path)
                txt_file = str(file).split('.')
                txt_file = f'{" ".join(txt_file[:-1])}.txt'
                new_file_path = os.path.join(save_path, txt_file)
                with open(new_file_path,'w',encoding = 'utf-8') as f:
                    f.write(doc.decode('utf-8'))
                
        return jsonify({'Uploaded':'True'})
    
    except Exception as e:
        return str(e), 500
    

@app.route('/createdb',methods = ['POST','GET'])
def createDB():
    json_ = request.json
    folder_path = json_['FolderPath']
    db_path = json_['DBPath']
    db_update = json_['UpdateDB']

    loader = DirectoryLoader(folder_path, glob='./*.txt', loader_cls=TextLoader,loader_kwargs = {'encoding':'utf-8'})
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 3000,chunk_overlap = 0)
    texts = text_splitter.split_documents(documents)

    if db_update == 'False':
        if os.path.exists(db_path):
            os.rmdir(db_path)
            os.makedirs(db_path)
        else:
            os.makedirs(db_path)
        
        embedding = OpenAIEmbeddings()

        vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding,
                                 persist_directory=db_path)
        
        vectordb.persist()
        return jsonify({'CreatedDB':'True'})

    elif db_update == "True":
        if not os.path.exists(db_path):
            os.makedirs(db_path)

            embedding = OpenAIEmbeddings()
            vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding,
                                 persist_directory=db_path)
            vectordb.persist()

        else:
            embedding = OpenAIEmbeddings()
            vectordb = Chroma(persist_directory = db_path,embedding_function = embedding)
            vectordb.add_documents(texts)
            vectordb.persist()
        
        return jsonify({'UpdatedDB':'True'})


def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])


@app.route('/getresumes',methods = ['POST','GET'])
def matchJD():
    json_ = request.json
    db_path = json_["DBPath"]
    job_description = json_["JobDescription"]

    prompt_template = """Use the following pieces of context, which corresponds to resumes of various individuals, and match these contexts with the job description given at the end.
    For each seperate resume given, there should be following output for each resume, the name of the individual if present in the resume, mobile number if it is present, the email if it is present, the domain of their resume, the domain of the job dsecription given, the percentage of match that the resume has with the job decription, and a reasoning for giving the match percentage.
    No need for making very deep matching percentage. If both the resume domain and job description domain are in similar fields but have different specializations, consider them as same. For example, if the resume domain is "Cybersecurity Penetration Tester", and job description domain is "Cybersecurtiy Analyst", though both are different specializations, consider them the same.
    If there are no matching resumes for the job description, just say that there are no matching resumes, don't try to make up an answer. Everything below the ###### line is an example of the expected input and output. Make use of the example and give appropriate answer as shown in the expected output of the example section given.
    Give the output for every piece of context given not just for a single context. All ontexts should contain the output as shown in the example.

    {context}

    Job Description: {question}

    ######
    Example of the inputs and outputs expeceted (Everything below this line is example, do not consider this while forming the output. Use this only as a reference for the output):

    John Doe, johndoe@hotmail.com, Software engineer, ABZ Tech, 12 years experience in developing application
    Duane Jones, dunaej@gmail.com, +1 323656362 ,Cybersecurity Analyst, XYC LLC. Cybersecurity analyst expert. Expereince in tools like SIEM.

    Job Description: We are looking for Cybersecurity analyst A Cybersecurity Analyst's responsibilities include reviewing computer networks and identifying any potential vulnerabilities, installing the necessary software in order to protect it from unauthorized access, and documenting detections so that future breaches can be mitigated efficiently.

    Expected Output:

    "Name": "John Doe",
    "Mobile": "None",
    "Email": "johndoe@hotmail.com",
    "Resume Domain": "Software Engineer",
    "Job Domain": "Cybersecurity Analyst",
    "Match Percentage": "25%",
    "Reasoning": <A GOOD MEANINGFUL REASONING>

    "Name": "Duane Jones",
    "Mobile": "+1 323656362",
    "Email": "duanej@gmail.com",
    "Resume Domain": "Cybersecurity Analyst",
    "Job Domain": "Cybersecurity ANalyst",
    "Match Percentage": "85%",
    "Reasoning": <A GOOD MEANINGFUL REASONING>

    ######

    As shown in the example, there are two resumes. The output also has all the required details for both the resumes. Same way your output should contain the details for all the resumes present in the context.
    """

    PROMPT = PromptTemplate(template = prompt_template,input_variables = ["context","question"])
    
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory = db_path,embedding_function = embedding)
    retriever = vectordb.as_retriever()

    chain_type_kwargs = {"prompt":PROMPT}
    qa = RetrievalQA.from_chain_type(llm = OpenAI(), chain_type = "stuff",chain_type_kwargs = chain_type_kwargs,retriever = retriever,
                                     return_source_documents = True)

    out = qa({"query":job_description})
    process_llm_response(out)
    return jsonify({"Matched":True})

if __name__ == '__main__':
    app.run(port = 5550)
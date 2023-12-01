from langchain import PromptTemplate
from langchain.llms import OpenAI
import os
from langchain.document_loaders import PyPDFLoader
import docx
os.environ["OPENAI_API_KEY"]="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"
def get_document(path):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    chapter = ""
    for page in pages:
        chapter += str(page)
    return chapter
def AI_call(legal_doc):
    template='''
    You are a helpfull assistant in  simplifying the provided legal documents in a user understandable language
    Simplify the text of provided EMPLOYMENT AGREEMENT legal document in a simplify and common understandable languagee in a legal format.
    Note: The response should be complete. and should cover the following points.
    1. Complete details of user and parties envolved.
    2. BY AND BETWEEN
    3. Employment: 
    4. Position Title: 
    5. Compensation
    6. Vacation: 
    7. Vacation: 
    8. Performance Reviews
    9. Obligations of the Employee
    10.	Intellectual Property Assignment
    11.	Confidentiality
    12.	Remedies
    13.	Termination
    14. Laws
    15.	Successors: 
    16.	Entire Agreement: 
    17. provided Context:
    18.	Severability: 
    IN WITNESS WHEREOF 

    The Given document is :-
    {text}'''
    prompt = PromptTemplate(input_variables=['text'],template=template)
    prompt = prompt.format(text=legal_doc)
    # get a chat completion from the formatted messages
    chat = OpenAI(temperature=0, model_name='gpt-3.5-turbo')
    response = chat(prompt)
    return response
content = get_document("Legal_documents_sample\EmployementAgreement.pdf")
sim_doc = AI_call(content)
# with open("sim_doc.txt","w") as file:
#     file.write(sim_doc)
# with open("sim_doc.txt","r") as file:
#      content = file.read()
doc = docx.Document()
doc.add_paragraph(sim_doc)

# Save the document
doc.save('generated_document.docx')

print("Word document created successfully.")
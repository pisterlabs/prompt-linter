import os
import json
import PyPDF2
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StdOutCallbackHandler

# Load the JSON file
with open('fragenkatalog.json', 'r', encoding='utf-8') as file:
    fragenkatalog = json.load(file)

# List all the files in the "Drucksachen" folder
drucksachen_folder = 'Drucksachen'
document_files = [f for f in os.listdir(drucksachen_folder) if f.endswith('.pdf')]

handler = StdOutCallbackHandler()
llm = ChatOpenAI(temperature=0, model='gpt-4-0314', streaming=True)

template = ChatPromptTemplate.from_messages([
    ("system", "Du bist juristischer Referent des Bundestages."),
    ("human", "Bitte beantworte diesen Fragenkatalog zu dem angehängten Dokument in angemessener Knappheit. Um die Fragen zu beantworten arbeite bitte in Stichpunkten.?"),
    ("ai", "Alles klar was sind die Fragen?"),
    ("human", "Die Fragen: {questions}. \n\n, Sei bitte so konkret wie möglich."),
    ("ai", "Okay, was ist das Dokument?"),
    ("human", "Das Dokument: {document}")
    ,
])

chain = LLMChain(llm=llm, prompt=template, callbacks=[handler])

# Process each document file
for document_file in document_files:
    # Determine the document type from the file name
    document_type, _ = os.path.splitext(document_file)

    # Retrieve the corresponding questions
    questions = fragenkatalog['DokumentTypen'].get(document_type)
    if questions is None:
        print(f'No questions found for document type: {document_type}')
        continue
    questions_str = '\n'.join(questions)

    # Load the document text
    document_path = os.path.join(drucksachen_folder, document_file)
    with open(document_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        document_text = ''
        for page_num in range(len(list(reader.pages))):
            page = reader.pages[page_num]
            document_text += page.extract_text()

    # Run the chain only specifying the input variables.
    result = chain.run({
        'document': document_text,
        'questions': questions_str
    })
    print(result)
    print("**********************")
    # Save the result to a file
    with open('results.txt', 'a') as f:
        f.write('******NEUES DOKUMENT*******\n')
        f.write(f'Document: {document_file}\n')
        f.write(f'Fragenkatalog für: {document_type}\n')
        f.write('Fragen:\n')
        f.write(questions_str)
        f.write('\n\LLM:\n')
        f.write(str(result))


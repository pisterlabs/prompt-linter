from langchain.document_loaders import ReadTheDocsLoader

#readopenaikey
def readopenaikey():
    with open('C:\\Users\\roshnipatil\\OneDrive - Microsoft\\GitHub_new\\openaikey.txt', 'r') as file:
        # Read all lines of the file
        return file.read()

#load documents 
def readPurviewDocs():
    loader = ReadTheDocsLoader('rtdocs')
    docs = loader.load()
    len(docs)

    print(docs[0].page_content)
    print(docs[5].page_content)
    docs[5].metadata['source'].replace('rtdocs/', 'https://')

    #Create a list of URL reference and page content
    data = []
    for doc in docs:
        data.append({
            'url': doc.metadata['source'].replace('rtdocs/', 'https://'),
            'text': doc.page_content
        })



#readpineconekey
def readpineconekey():
    with open('C:\\Users\\roshnipatil\\OneDrive - Microsoft\\GitHub_new\\pineconekey.txt', 'r') as file:
        # Read all lines of the file
        lines = file.readlines()
        # Print the content of the file
        for line in lines:
            print(line)


#Open the file for reading
def readSampleDoc():
    with open('C:\\Users\\roshnipatil\\OneDrive - Microsoft\\GitHub_new\\testpii.txt', 'r') as file:
        # Read all lines of the file
        return file.read()


#Open the file for reading
def writeprompttofile(prompt_text):
    with open('C:\\Users\\roshnipatil\\OneDrive - Microsoft\\GitHub_new\\search_prompts.txt', 'w') as file:
        file.write(prompt_text)


#Open the file for reading
def writesencheckprompttofile(sen_prompt_text):
    with open('C:\\Users\\roshnipatil\\OneDrive - Microsoft\\GitHub_new\\sen_check_prompts.txt', 'w') as file:
        file.write(sen_prompt_text)


#Open the file for reading
def writesenlistprompttofile(sen_prompt_text):
    with open('C:\\Users\\roshnipatil\\OneDrive - Microsoft\\GitHub_new\\sen_list_prompts.txt', 'w') as file:
        file.write(sen_prompt_text)

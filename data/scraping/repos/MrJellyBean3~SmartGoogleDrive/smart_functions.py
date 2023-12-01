# Drive Imports
import yaml
import asyncio
from deferred_imports import langchain, imports_done
import webbrowser

# Global Variables
dictionaries_folder_path=""
structure_dictionary_path=""
information_dictionary_path=""
folder_dictionary_path=""



# Information Mapping
async def a_update_mapping(your_dictionary,override=False):
    tasks=[]
    for key in list(your_dictionary.keys()):
        if override or 'mappedDate' not in your_dictionary[key] or not your_dictionary[key]['mappedDate'] or your_dictionary[key]['modifiedDate'] > your_dictionary[key]['mappedDate']:
            tasks.append(a_generate_mapping(your_dictionary[key]['content'],your_dictionary[key]['title'],your_dictionary[key]['path'],key))

    results=await asyncio.gather(*tasks)
    for item in results:
        id=list(item.keys())[0]
        your_dictionary[id]['mapping'] = item[id]
        your_dictionary[id]['mapped']=True
        your_dictionary[id]['mappedDate'] = your_dictionary[id]['modifiedDate']

    return your_dictionary



# Information Mapping
async def a_generate_mapping(content,title,parent,id):
    imports_done.wait()
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    # Define the templates
    system_template="I want you to give a summary of the of the document the user gives you as if you were describing it and what it is for, the user will also tell you the path of the parent directory of the document and its title which you should use to understand what the summary should be, for example a 'book summaries' document's summary should include that it is a summary of books and not simply the contents of the books."
    human_template="Here is my document title-{title}, with parent directory-{parent}, and here is content of the document:\n{content}"
    system_template2="""You are to provide a useful summary and description so that someone reading your description would know what you are refering to. Use the context of the title and parent directory to understand what the description and summary should be. Think before begin to describe the document. Break down the path into sections splitting with each \\ and think about them out loud, tell me what the meaning of each directory is in your interpretation. After you are finished thinking give your description . Your response should follow this template:
    \"'Thoughts: 'what your thoughts are on the meaning of the document are and what it is for with relation to its title and parent directory'
    Description: 'your description and summary based on your thoughts so someone would know what this is for'\""""

    # Create the prompt
    system_message=SystemMessagePromptTemplate.from_template(system_template)
    human_message=HumanMessagePromptTemplate.from_template(human_template)
    system_message2=SystemMessagePromptTemplate.from_template(system_template2)
    message_list=[system_message,human_message,system_message2]
    chat_prompt=ChatPromptTemplate.from_messages(message_list)


    # Generate the mapping
    formated_prompt=chat_prompt.format_prompt(content=content, title=title, parent=parent).to_messages()
    raw_string_prompt=""
    for item in formated_prompt:
        raw_string_prompt+=item.type+": "+item.content+"\n\n"
    if len(raw_string_prompt)>9000:
        model_name="gpt-3.5-turbo-16k"
    else:
        model_name="gpt-3.5-turbo"
    chat=ChatOpenAI(model=model_name,temperature=0.3)
    chat_response=await chat.agenerate([formated_prompt])
    print(title+" "+parent+" "+chat_response.generations[0][0].text+"\n\n")
    output_string=chat_response.generations[0][0].text

    # Parse the mapping
    mapped_result=""
    mapped_result=(output_string).split("Description: ")[-1]
    if mapped_result=="":
        mapped_result=(output_string).split("Description:")[-1]
    return {id:mapped_result}





# Folder Mapping
async def a_update_folder_mapping(folder_dictionary,information_dictionary,override=False):
    finished_folders=[]
    length_folders=len(list(folder_dictionary.keys()))
    results=[]
    while(length_folders>len(finished_folders)):
        tasks=[]
        print("finished folders: "+str(len(finished_folders))+"/"+str(length_folders))
        for key in list(folder_dictionary.keys()):
            #check if the key is already mapped
            if not override and 'mappedDate' in folder_dictionary[key] and folder_dictionary[key]['mappedDate'] and folder_dictionary[key]['modifiedDate'] <= folder_dictionary[key]['mappedDate']:
                finished_folders.append(key)
                print("Already done: "+key)
            else:
                print("Not done: "+key, override, 'mappedDate' in folder_dictionary[key], folder_dictionary[key]['mappedDate'], folder_dictionary[key]['modifiedDate'] <= folder_dictionary[key]['mappedDate'])

            if key not in finished_folders:
                if folder_dictionary[key]["contained_folder_ids"]==[]:
                    #Create task
                    contents=""
                    for file_id in folder_dictionary[key]["contained_file_ids"]:
                        contents+=(information_dictionary[file_id]["mapping"])+"\n"
                    for folder_id in folder_dictionary[key]["contained_folder_ids"]:
                        contents+=(folder_dictionary[folder_id]["mapping"])+"\n"
                    tasks.append(a_generate_folder_mapping(contents,folder_dictionary[key]['title'],folder_dictionary[key]['path'],key))
                    finished_folders.append(key)
                else:
                    all_completed=True
                    for cf in folder_dictionary[key]["contained_folder_ids"]:
                        if cf not in finished_folders:
                            all_completed=False
                    if all_completed:
                        #Create task
                        contents=""
                        for file_id in folder_dictionary[key]["contained_file_ids"]:
                            contents+=(information_dictionary[file_id]["mapping"])+"\n"
                        for folder_id in folder_dictionary[key]["contained_folder_ids"]:
                            contents+=(folder_dictionary[folder_id]["mapping"])+"\n"
                        tasks.append(a_generate_folder_mapping(contents,folder_dictionary[key]['title'],folder_dictionary[key]['path'],key))
                        finished_folders.append(key)
        results.append(await asyncio.gather(*tasks))

    for result in results:
        for item in result:
            id=list(item.keys())[0]
            folder_dictionary[id]['mapping'] = item[id]
            folder_dictionary[id]['mapped']=True
            folder_dictionary[id]['mappedDate'] = folder_dictionary[id]['modifiedDate']
    return(folder_dictionary)




# Folder Mapping
async def a_generate_folder_mapping(contents,title,parent,id):
    # Setup imports
    imports_done.wait()
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )

    # Define the templates
    system_template="I want you to give a summary of the of the folder the user gives you as if you were describing it and what it is for, the user will also tell you the path of the parent directory of the folder, the folder title, and the descriptions of the contents of the files or folders the folder contains which you should use to understand what the description should be."
    human_template="Here is my folder title-{title}, with parent directory-{parent}, and here are the contents of the folder:\n{contents}"
    system_template2="""You are to provide a useful summary and description so that someone reading your description would know what you are refering to. Use the context of the title and parent directory to understand what the description and summary should be. Think before you begin to describe the document. Break down the path into sections splitting with each \\ and think about them out loud, tell me what the meaning of each directory is in your interpretation. After you are finished thinking give your description . Your response should follow this template:
    \"'Thoughts: 'what your thoughts are on the meaning of the folder are and what it is for with relation to its title and parent directory'
    Description: 'your description and summary based on your thoughts so someone would know what this is for'\""""

    # Create the prompt
    system_message=SystemMessagePromptTemplate.from_template(system_template)
    human_message=HumanMessagePromptTemplate.from_template(human_template)
    system_message2=SystemMessagePromptTemplate.from_template(system_template2)
    message_list=[system_message,human_message,system_message2]
    chat_prompt=ChatPromptTemplate.from_messages(message_list)

    # Get the response of the mapping for the item
    formated_prompt=chat_prompt.format_prompt(contents=contents, title=title, parent=parent).to_messages()
    raw_string_prompt=""
    for item in formated_prompt:
        raw_string_prompt+=item.type+": "+item.content+"\n\n"
    if len(raw_string_prompt)>9000:
        model_name="gpt-3.5-turbo-16k"
    else:
        model_name="gpt-3.5-turbo"
    chat=ChatOpenAI(model=model_name,temperature=0.3)
    chat_response=await chat.agenerate([formated_prompt])
    print(title+" "+parent+" "+chat_response.generations[0][0].text+"\n\n")
    output_string=chat_response.generations[0][0].text

    # Parse the mapping
    mapped_result=(output_string).split("Description: ")[-1]
    if mapped_result=="":
        mapped_result=(output_string).split("Description:")[-1]
    return {id:mapped_result}




# Generate Mappings
def map(override=False):
    # Setup dictionary paths
    global dictionaries_folder_path, structure_dictionary_path, information_dictionary_path, folder_dictionary_path

    # map information dictionary
    with open(information_dictionary_path, "r") as file:
            information_dict = yaml.load(file, Loader=yaml.FullLoader)
    information_dict=asyncio.run(a_update_mapping(information_dict,override=override))
    with open(information_dictionary_path, 'w') as outfile:
        yaml.dump(information_dict, outfile)

    # Map the folder dictionary
    with open(information_dictionary_path, "r") as file:
            information_dict = yaml.load(file, Loader=yaml.FullLoader)
    with open(folder_dictionary_path, "r") as file:
        folder_dictionary = yaml.load(file, Loader=yaml.FullLoader)
    folder_dictionary=asyncio.run(a_update_folder_mapping(folder_dictionary,information_dict,override=False))
    with open(folder_dictionary_path, 'w') as outfile:
        yaml.dump(folder_dictionary, outfile)
    print("Done mapping")




# Update Database
def update_vectordb(persist_directory,finish_que):
    # Setup imports
    imports_done.wait()
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings

    # Create custom Document class
    class Document:
        def __init__(self, page_content="",source="",dict_id="",mimeType="",title=""):
            self.page_content = page_content
            self.metadata={'source': source, 'id': dict_id, "mimeType":mimeType,"title":title}
        def __repr__(self):
            attributes = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
            return f"Document({attributes})"

    # Read from information dictionary
    global dictionaries_folder_path, structure_dictionary_path, information_dictionary_path, folder_dictionary_path
    if "information" in persist_directory:
        base_dict_file_name=information_dictionary_path
    elif "folder" in persist_directory:
        base_dict_file_name=folder_dictionary_path
    with open(base_dict_file_name) as f:
        base_dict = yaml.load(f, Loader=yaml.FullLoader)

    # Create list of documents
    my_documents = []
    for key in list(base_dict.keys()):
        if base_dict[key]["path"]=="":
            my_documents.append(Document(base_dict[key]["mapping"],source=base_dict[key]["path"]+"none id:"+base_dict[key]["id"]+":mimeType:"+base_dict[key]["mimeType"], dict_id=base_dict[key]["id"],mimeType=base_dict[key]["mimeType"],title=base_dict[key]["title"]))
        else:
            my_documents.append(Document(base_dict[key]["mapping"],source=base_dict[key]["path"]+" id:"+base_dict[key]["id"]+":mimeType:"+base_dict[key]["mimeType"], dict_id=base_dict[key]["id"],mimeType=base_dict[key]["mimeType"],title=base_dict[key]["title"]))
    
    # Delete and regenerate the database
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    try:
        vectordb.delete_collection()
        vectordb.persist()
    except Exception as e:
        print(e)
    vectordb = Chroma.from_documents(
        documents=my_documents, 
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectordb.persist()
    vectordb = None

    # Depricated queue usage
    finish_que.put(True)




# Make Vector Database with files and folders
def combine_vectordb(persist_directory):
    #Setup imports and paths
    imports_done.wait()
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings
    global dictionaries_folder_path, structure_dictionary_path, information_dictionary_path, folder_dictionary_path
    
    # Create custom Document class
    class Document:
        def __init__(self, page_content="",source="",dict_id="",mimeType="",title=""):
            self.page_content = page_content
            self.metadata={'source': source, 'id': dict_id, "mimeType":mimeType,"title":title}
        def __repr__(self):
            attributes = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
            return f"Document({attributes})"
        
    # Dicitonary to list of documents function
    def add_documents(base_dict):
        my_documents = []
        for key in list(base_dict.keys()):
            if base_dict[key]["path"]=="":
                my_documents.append(Document(base_dict[key]["mapping"],source=base_dict[key]["path"]+"none id:"+base_dict[key]["id"]+":mimeType:"+base_dict[key]["mimeType"], dict_id=base_dict[key]["id"],mimeType=base_dict[key]["mimeType"],title=base_dict[key]["title"]))
            else:
                my_documents.append(Document(base_dict[key]["mapping"],source=base_dict[key]["path"]+" id:"+base_dict[key]["id"]+":mimeType:"+base_dict[key]["mimeType"], dict_id=base_dict[key]["id"],mimeType=base_dict[key]["mimeType"],title=base_dict[key]["title"]))
        return(my_documents)

    # Read from information and folder dictionaries
    with open(information_dictionary_path) as f:
        information_dict = yaml.load(f, Loader=yaml.FullLoader)
    with open(folder_dictionary_path) as f:
        folder_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    # Turn dictionaries into document list
    my_documents=add_documents(information_dict)+add_documents(folder_dict)

    # Delete and regenerate the combined_db database
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    try:
        vectordb.delete_collection()
        vectordb.persist()
    except Exception as e:
        print(e)
    vectordb = Chroma.from_documents(
        documents=my_documents, 
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectordb.persist()
    vectordb = None
    print("Finished combining databases")




# Retrieve From Information
def retrieve_from_information(user_question,return_que):
    # Setup imports
    imports_done.wait()
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings

    # Get vectordb
    persist_directory = 'combined_db'
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    
    # Retrive documents
    docs_and_scores = vectordb.similarity_search_with_score(user_question)

    # Get docs from docs_and_scores
    docs=[]
    scores=[]
    for item in docs_and_scores:
        docs.append(item[0])
        scores.append(item[1])
    print(scores)

    # Open the website for the first doc
    open_website(docs[0])

    # Pass the docs to the main app
    return_que.put(docs)




# Retrieve From Folder
def retrieve_from_folder(user_question):
    # Setup imports
    imports_done.wait()
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings

    # Get vectordb
    persist_directory = 'folder_db'
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    # Retrive documents
    docs_and_scores = vectordb.similarity_search_with_score(user_question)

    # Get docs from docs_and_scores
    docs=[]
    scores=[]
    for item in docs_and_scores:
        docs.append(item[0])
        scores.append(item[1])
    print(scores)

    # Open the website for the first doc
    open_website(docs[0])



# Opens website of a document
def open_website(doc):
    if doc==None:
        print("No documents found")
    else:
        url=None
        if "spreadsheet" in doc.metadata["mimeType"]:
            url = "https://docs.google.com/spreadsheets/d/"+doc.metadata["id"]
        elif "document" in doc.metadata["mimeType"]:
            url = "https://docs.google.com/document/d/"+doc.metadata["id"]
        elif "folder" in doc.metadata["mimeType"]:
            url = "https://drive.google.com/drive/folders/"+doc.metadata["id"]
        print(url)
        if url != None:
            webbrowser.open(url)


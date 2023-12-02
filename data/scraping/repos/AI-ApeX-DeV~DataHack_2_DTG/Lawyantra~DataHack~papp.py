# from langchain.document_loaders import CSVLoader
# from langchain.indexes import VectorstoreIndexCreator
# from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI
# import os
# import pickle
# import sqlite3

# os.environ["openai_api_key"] = "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"

# loader = CSVLoader(file_path='updated_lawyer_data.csv')

# index_creator = VectorstoreIndexCreator()
# docsearch = index_creator.from_loaders([loader])

# chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")

# def copy_chain_without_sqlite_connections(chain):
#     # Helper function to recursively exclude sqlite3.Connection objects
#     def exclude_sqlite_connections(obj):
#         if isinstance(obj, sqlite3.Connection):
#             return None
#         if isinstance(obj, dict):
#             return {k: exclude_sqlite_connections(v) for k, v in obj.items() if exclude_sqlite_connections(v) is not None}
#         if isinstance(obj, list):
#             return [exclude_sqlite_connections(item) for item in obj if exclude_sqlite_connections(item) is not None]
#         return obj

#     return exclude_sqlite_connections(chain)

# # Use custom serialization to save the trained model
# with open('retrieval_qa_model.pkl', 'wb') as model_file:
#     serializable_chain = copy_chain_without_sqlite_connections(chain)
#     pickle.dump(serializable_chain, model_file)

# When you want to use the model for inference:
# Load the model from the saved file
# with open('retrieval_qa_model.pkl', 'rb') as model_file:
#     loaded_chain = pickle.load(model_file)

# Get a user query
# query = input("Enter your query: ")

# # Perform inference using the loaded model
# response = loaded_chain({"question": query })

# # Print the response
# print(response['result'])


from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
import pickle

os.environ["openai_api_key"] = "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"

loader = CSVLoader(file_path='PS_2_Test_Dataset.csv')

index_creator = VectorstoreIndexCreator()
docsearch = index_creator.from_loaders([loader])

chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")
    
query="Two brothers were tenant of a landlord in a commercial property.One brother had one son and adaughter (both minor) when he got divorced with his wife.The children's went into mother's custody at thetime of divorce and after some years the husband (co tenant) also died. Now can the children of thedeceased brother(co tenant) claim the right. which lawyer should be helpful for me?"
response =  chain({"question":query})

print(response['result'])

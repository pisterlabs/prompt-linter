# from langchain import OpenAI


# Slightly less than 0.1 USD to build the index this way
# documents = SimpleDirectoryReader("./data").load_data()
# # Need api key here
# index = GPTSimpleVectorIndex(documents)

# print(index)
# index.save_to_disk("index.json")
# Pass in query and get response
# index = GPTSimpleVectorIndex.load_from_disk("index.json")
# response = index.query("Who is Anuraag?")
# print(response)


# set maximum input size
# max_input_size = 4096
# # set number of output tokens
# num_outputs = 256
# # set maximum chunk overlap
# max_chunk_overlap = 20
# # set chunk size limit
# chunk_size_limit = 600

# # define LLM (ChatGPT gpt-3.5-turbo)
# llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
# prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

# documents = SimpleDirectoryReader("./data").load_data()

# index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

# index.save_to_disk("index-2.json")


# print(response)


# @app.post("/create-index")
# def create_index(file: UploadFile = Form("file")):
#     try:
#         file_location = f"./data/{file.filename}"
#         with open(file_location, "wb+") as file_object:
#             shutil.copyfileobj(file.file, file_object)

#         MY_TEXT = docx2txt.process(file_location)
#         with open("./output/output.txt", "w") as text_file:
#             print(MY_TEXT, file=text_file)

#         documents = SimpleDirectoryReader("./output").load_data()
#         print(documents)
#         # llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
#         # prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
#         # index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
#     except Exception:
#         return {"message": "There was an error uploading the file"}
#     finally:
#         file.file.close()

#     return {"message": f"Successfully uploaded {file.filename}"}


# @app.get('/update-index')
# async def update_index():
#     # documents = SimpleDirectoryReader("./data").load_data()
#     # print(documents)
#     # llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
#     # prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
#     # index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
#     # index.save_to_disk('./index-long.json')  
#     # 6106 tokens
#     documents = SimpleDirectoryReader("./data").load_data()
#     print(documents)
#     llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.3, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
#     prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
#     index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
#     index.save_to_disk('./index-long-003.json')  
#     # 6106 tokens

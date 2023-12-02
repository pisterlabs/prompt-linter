from llama_index import SimpleDirectoryReader, GPTListIndex, LLMPredictor, ServiceContext
from langchain import OpenAI
import os
import sys
from datetime import datetime


def construct_index_from_file(file_path: str):
    num_outputs = 512
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
    index = GPTListIndex.from_documents(docs, service_context=service_context)
    return index


def process_one_file(in_filepath: str) -> bool:
    out_filepath = f"{'.'.join(in_filepath.split('.')[0:-1])}.bd2"
    if os.path.exists(out_filepath):
        return False
    else:
        index = construct_index_from_file(in_filepath)
        query_engine = index.as_query_engine()
        print(datetime.now())
        response = query_engine.query(
            "What are the business domains this source code is about? Please answer only with a list of business domains!")
        print(response)
        with open(out_filepath, "w") as out:
          out.write(str(response))
          print(f"--> {response}")


def process_folder(folder_path: str):
  counter = 0
  for subdir, dirs, files in os.walk(folder_path):
      for file in files:
        if file.endswith("py.bd"):
          file_abs_path = subdir + os.path.sep + file
          processed = process_one_file(file_abs_path)
          counter += 1
          if processed:
            print(counter)
            if counter > 1000:
              sys.exit(0)


process_folder("/Users/oliverwidder/dev/github/nlp/dict/_all/antlr/words")

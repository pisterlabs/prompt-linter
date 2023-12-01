from gpt_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import sys
import os
from IPython.display import Markdown, display

os.environ["OPENAI_API_KEY"] = "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"

'''
这段代码实现了一个简单的问答系统，可以利用 OpenAI 的 API 进行自然语言处理和生成，以回答用户输入的问题。

首先，construct_index() 函数将指定目录中的所有文本文件读取到内存中，然后使用 GPT 模型生成每个文本的向量表示，并使用这些向量构建索引。这个索引将被保存到磁盘上，以便以后使用。

接下来，ask_ai() 函数会加载保存在磁盘上的索引，并进入一个循环，等待用户输入问题。当用户输入问题后，系统会将该问题与索引中的所有文本向量进行比较，并使用 LLM 模型计算每个文本向量对该问题的响应。然后，系统将选择具有最佳响应的文本向量，并使用 GPT 模型生成一个完整的响应。最后，该响应将显示在屏幕上。

'''



def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 300
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index.save_to_disk('index.json')
    print("Done.")
    return index


def ask_ai():
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    while True:
        query = input("What do you want to ask? ")
        response = index.query(query, response_mode="compact")
        print(f"Response: {response}")
        # display(Markdown(f"Response: <b>{response.response}</b>"))

if __name__ == '__main__':
    ask_ai()
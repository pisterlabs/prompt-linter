from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models.openai import ChatOpenAI
from datetime import datetime

def web_qa(url_list, query, out_name):
    openai = ChatOpenAI(
        model_name="gpt-4",
        max_tokens=2048
    )
    loader_list = []
    for i in url_list:
        print('loading url: %s' % i)
        loader_list.append(WebBaseLoader(i))

    index = VectorstoreIndexCreator().from_loaders(loader_list)
    ans = index.query(question=query,
                      llm=openai)
    print("")
    print(ans)

    outfile_name = out_name + datetime.now().strftime("%m-%d-%y-%H%M%S") + ".out"
    with open(outfile_name, 'w') as f:
        f.write(ans)

url_list = [
    "https://openaimaster.com/how-to-use-ideogram-ai/",
    "https://dataconomy.com/2023/08/28/what-is-ideogram-ai-and-how-to-use-it/",
    "https://ideogram.ai/launch",
    "https://venturebeat.com/ai/watch-out-midjourney-ideogram-launches-ai-image-generator-with-impressive-typography/"
]

prompt = '''
    Given the context, please provide the following:
    1. summary of what it is
    2. summary of what it does
    3. summary of how to use it
    4. Please provide 5 interesting prompts that could be used with this AI.
'''

web_qa(url_list, prompt, "summary")

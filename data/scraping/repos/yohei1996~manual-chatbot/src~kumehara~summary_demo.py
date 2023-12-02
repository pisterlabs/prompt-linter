from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
import gradio as gr
import pickle
import os



with open("../../storage/kumeharadocuments/vectorstore_kume__200_50.pkl", "rb") as f:
    vectorstore:FAISS = pickle.load(f)

retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

template = """
### 命令文
あなたは有名な編集者です。以下の制約をもとに著者になりきって文章を書いてください。

### 制約
- [theme:]について[reference text:]の内容をもとに書いてください
- 100文字以内の文章を書いてください。
- 結果を[### output] セクションの <answer here> に出力してください。
- [reference text:]の中から参考にしたものを[### output] セクションの <refarence sources here> に出力してください。

### リソース
theme: {question}

reference text:
'''
{reference_text}
'''
### output
<answer here>
参考:<refarence sources here>
"""


def chat(message, history=[]):
    try:
        llm = ChatOpenAI(model_name= "gpt-3.5-turbo")
        prompt = PromptTemplate(
            input_variables=["question","reference_text"],
            template=template,
        )
        docs = retriever.get_relevant_documents(message)
        reference_text = ""
        for doc in docs:
            if len(reference_text) > 300: break
            reference_text += f'text:\n{doc.page_content}\nsource:\n{doc.metadata["source"]}\n\n'
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run({"question":message,"reference_text":reference_text})
    except Exception as e:
        response = f"予期しないエラーが発生しました: {e}"

    history.append((message, response))
    return history, history

chatbot = gr.Chatbot()

demo = gr.Interface(
    chat,
    ['text',  'state'],
    [chatbot, 'state'],
    allow_flagging = 'never',
)

demo.launch()
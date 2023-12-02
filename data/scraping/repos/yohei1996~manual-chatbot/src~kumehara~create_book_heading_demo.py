
import os
from langchain.document_loaders import TextLoader
import MeCab
import re
from collections import Counter
import openai
import time
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
import gradio as gr


def save(text,path):
    with open(path, "w") as file:
        file.write(text)
def read(path):
    with open(path, "r") as file:
        return file.read()
    
mecab = MeCab.Tagger()

def Mecab(K_text):
    K_parsed = mecab.parse(K_text)
    K_parsed_lines = K_parsed.split('\n')
    #処理に使うリストを作成
    K_parsed_words = []
    K_words = []
    
    for K_parsed_line in K_parsed_lines:
        K_parsed_words.append(re.split('[\t,]', K_parsed_line))
    
    #名詞・一般に該当する単語をリストに格納
    for K_parsed_word in K_parsed_words:
        if (    
            K_parsed_word[0] not in ('EOS', '') 
            # and K_parsed_word[2] in  ['副詞可能'] #フィルター用
            and K_parsed_word[1] in  ['名詞'] 
            and K_parsed_word[2] in ['一般','自立','サ変接続']
            and K_parsed_word[7] not in [
                '）',
                '（',
                'する',
                'できる',
                '言う',
                'いう',
                'ある',
                'なる',
                '分かる',
                'わかる',
                '聞く',
                '思う',
                '.'
                ]
            ):
            if K_parsed_word[1] == '動詞':
                K_words.append(K_parsed_word[7])
            else:
                K_words.append(K_parsed_word[0])

    K_counter = Counter(K_words)
    words=[K_word for K_word, K_count in K_counter.most_common()]
    return ",".join(words)

def save_all_docs():
    docs = ""
    doc_dir = '../../storage/kumeharadocuments'
    for filename in os.listdir(doc_dir):
        filepath = os.path.join(f'{doc_dir}', filename)
        loader = TextLoader(filepath)
        file = loader.load()[0]
        doc = file.page_content
        meta_data = file.metadata
        docs = docs + doc.strip()

    save(docs, "../../storage/kumeharadocuments/all_docs.txt")

def translate_to_mecab():
    docs = read("../../storage/kumeharadocuments/all_docs.txt")
    mecab_docs = Mecab(docs)
    save(mecab_docs, "../../storage/kumeharadocuments/all_docs_mecab.txt")

def createHeadings():
    template = """
    ### 命令文
    あなたは有名な編集者です。以下の制約をもとに本の見出しを書いてください。

    ### 制約
    - [{question}]というタイトルの本の見出しを書いてください。
    - 大見出しを5個と各大見出しの中に中見出しを3個それぞれ書いてください。
    - 本の見出しは、[reference text:]の単語をなるべく用いて作成してください。
    - [output format:]の単語をなるべく用いて作成してください。
    - 流れがわかるように、大見出しと小見出しを適切に並べてください。

    ### リソース
    theme: {question}

    reference text:
    '''
    {reference_text}
    '''
    ### output format
    -  <ここに大見出し>
        ┗ <ここに中見出し>
        ┗ <ここに中見出し>
        ┗ <ここに中見出し>
        ...
    - <ここに大見出し>
        ┗ <ここに中見出し>
        ┗ <ここに中見出し>
        ┗ <ここに中見出し>
    
    
    """

    prompt = PromptTemplate(
                input_variables=["question","reference_text"],
                template=template,
    )

    reference_text = read("../../storage/kumeharadocuments/all_docs_mecab.txt")

    def chat(message, history=[]):
        try:
            llm = ChatOpenAI(model_name= "gpt-3.5-turbo")
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



# save_all_docs()
# translate_to_mecab()
createHeadings()
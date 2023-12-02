import pandas as pd
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

if __name__ == "__main__":
    # 学習用データを読込
    df = pd.read_csv("faq.csv")
    # 前処理
    # 余分な改行を削除
    # df = df.replace("\n\n\n", "", regex=True)
    # 全角スペースを半角スペースに変換
    # df = df.replace("\u3000", " ", regex=True)

    # 学習用データをLangChainで使えるようにする
    docs = []
    for index, row in df.iterrows():
        question = row["question"]
        answer = row["answer"]
        page_content = f"""question: {question}m\nanswer: {answer}"""
        docs.append(Document(page_content=page_content))

    # 学習用データが長文の場合はテキストを分割する
    # splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # sub_docs = splitter.split_documents(docs)

    # ベクトルデータベースをローカルに保存
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faq_index")
    retriever = db.as_retriever()

    # プロンプトと会話履歴
    system_template = """あなたはKANNAのカスタマーサポートです。丁寧な回答を心がけてください。以下のContextを使用して質問にのみ答えてください。答えがわからない場合は、「わかりません」と回答してください。

Context:
{context}"""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain_type_kwargs = {"prompt": prompt}
    # TODO temperatureを変更してみましょう
    # TODO gpt-3.5-turboの挙動も確認してみましょう
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, model_name="gpt-4"),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
    )
    question = "どの端末から入力できますか？それと、クラウド上にあるデータはダウンロードすることができますか？"
    # TODO 学習データに存在しない答えについてテストしてみましょう
    # question = "ペアケアについて教えて下さい"
    answer = qa.run(question)
    print(answer)

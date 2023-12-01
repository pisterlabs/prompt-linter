import torch
import sentence_transformers
from chatglm_llm import ChatGLM
from pymilvus import connections, Collection
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModel

connections.connect(
    alias="default",
    host='localhost',
    port='19530'
)

collection = Collection("document")  # Get an existing collection.
collection.load()
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")


# 获取文本的向量
def get_vector(text):
    input_ids = tokenizer(text, padding=True, truncation=True, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        output = model(input_ids)[0][:, 0, :].numpy()
    return output.tolist()[0]


# 定义查询函数
def search_similar_text(input_text):
    # 将输入文本转换为向量
    input_vector = get_vector(input_text)

    similarity = collection.search(
        data=[input_vector],
        anns_field="vector",
        param={"metric_type": "IP", "params": {"nprobe": 10}, "offset": 0},
        limit=3,
        expr=None,
        consistency_level="Strong"
    )
    ids = similarity[0].ids
    res = collection.query(
        expr=f"id in {ids}",
        offset=0,
        limit=1,
        output_fields=["id", "content", "title"],
        consistency_level="Strong"
    )
    return res


# 通过AI模型获取回答
def get_answer(input_text, reference_data):
    documents = [
        Document(page_content=i['title'] + '\n' + i['content'], metadata={"source": ""}) for i in reference_data
    ]

    prompt_template = """
    基于以下已知信息，简洁和专业的来回答用户的问题。
    如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。

    已知内容:
    {context}

    问题:
    {question}
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    embeddings = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese", )
    embeddings.client = sentence_transformers.SentenceTransformer(embeddings.model_name,
                                                                  device=DEVICE)
    vector_store = Milvus.from_documents(documents, embeddings)
    chatglm = ChatGLM()
    chatglm.load_model(
        model_name_or_path="THUDM/chatglm-6b"
    )
    chatglm.history_len = 3
    # chatglm.history = []
    knowledge_chain = RetrievalQA.from_llm(
        llm=chatglm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 6}),
        prompt=prompt
    )
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}"
    )

    knowledge_chain.return_source_documents = True

    result = knowledge_chain({"query": input_text})
    # chatglm.history[-1][0] = input_text
    print(result)


if __name__ == "__main__":
    # question = input('Please enter your question: ')
    question = "怎样在CRM中设置礼品卡"

    res = search_similar_text(question)
    get_answer(question, res)

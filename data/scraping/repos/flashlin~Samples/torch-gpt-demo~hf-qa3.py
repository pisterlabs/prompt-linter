import torch
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from pdf_utils import splitting_documents_into_texts, load_txt_documents_from_directory
from vectordb_utils import load_chroma_from_documents, MyEmbeddingFunction

model_name = "deepset/bert-base-cased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)


def get_embed_text(text: str):
    # input_ids = tokenizer.encode(text, max_length=512, truncation=True, return_tensors="pt")
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]
    embedding = torch.mean(last_hidden_states, dim=1).numpy()
    return embedding


documents = load_txt_documents_from_directory('./news')
texts = splitting_documents_into_texts(documents)

embedding_function = MyEmbeddingFunction(get_embed_text)
vectordb = load_chroma_from_documents(texts, embedding_function)


class CustomChroma(Chroma):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def search(self, question, k=3):
        # 使用問題文本 'question' 進行搜尋，並返回最相關的文件
        # query_embedding = self._embedding_function.embed_query(question)
        query_embedding = get_embed_text(question)
        similar_docs = self.similarity_search(query_embedding, k=k)
        return similar_docs


vectordb = CustomChroma.from_documents(documents=texts, embedding_function=embedding_function)
# 創建自訂的 QA retriever
retriever = VectorStoreRetriever(vectorstore=vectordb)

# retriever = vectordb.as_retriever(search_kwargs={"k": 10})


def answer_question(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    score = outputs[1][0, answer_start].item()
    return answer, score

question = "WSL2 is very slow, How to resolve it?"
docs = retriever.get_relevant_documents(question)

best_doc = ''
best_answer = ''
best_score = 0
for doc in docs:
    answer, score = answer_question(question, doc.page_content)
    print(f'{doc.page_content=} {answer=} {score=}')
    print('')
    if score < best_score:
        best_answer = answer
        best_score = score
        best_doc = doc.page_content

print('')
print(f'最佳回答: {best_answer}')
print(f'{best_doc}')

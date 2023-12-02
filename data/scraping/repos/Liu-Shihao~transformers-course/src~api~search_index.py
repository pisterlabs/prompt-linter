from flask import Flask, request, jsonify
from langchain import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

app = Flask(__name__)

# research

embeddings = HuggingFaceEmbeddings()
db1 = FAISS.from_texts(["foo"], embeddings)
retriever = db1.as_retriever(search_kwargs={"k": 2})

'''
curl --location --request GET 'http://127.0.0.1:5000/search?q=好'
'''
@app.route('/search', methods=['GET'])
def search_index():
    query = request.args.get('q')
    docs = retriever.get_relevant_documents(query)
    return str(docs)

'''
curl --location --request POST 'http://127.0.0.1:5000/push' \
--header 'Content-Type: application/json' \
--data-raw '{
    "body": "你好"
}'
'''
@app.route('/push', methods=['POST'])
def push_index():
    data = request.get_json()
    # 从JSON参数中获取特定的值
    # name = data['name']
    # age = data['age']
    print(data)
    db2 = FAISS.from_texts([data['body']], embeddings)
    db1.merge_from(db2)
    print(db1.docstore._dict)
    response = {
        "code": "200",
        "msg": "Success",
        "data": str(db1.docstore._dict)
    }
    return jsonify(response), 200


if __name__ == '__main__':
    app.run()
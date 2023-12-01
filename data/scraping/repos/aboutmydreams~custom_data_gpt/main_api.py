from flask import Flask, jsonify, request
from typing import List, Union
import requests
import faiss
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle, os  # noqa: E401
from langchain import OpenAI, LLMChain
from langchain.prompts import Prompt
from dotenv import load_dotenv

from utils.flask_input_fixed import Inputs, JsonSchema
from utils.get_file_size import get_file_size
from utils.translate import get_language_type, translate

load_dotenv()
app = Flask(__name__)


class TrainInputs(Inputs):
    """
    针对 '/train' 视图函数的输入参数规则
    """

    json = [
        JsonSchema(
            {
                "space_name": Union[str, int],
                "file_list": List[str],
                "file_tag": str,  # "url" | "path"
            }
        )
    ]


@app.route("/train", methods=["POST"])
def train():
    # 使用我们定义的 TrainInputs 类进行参数验证，如果有错误则返回 400 状态码及错误信息
    inputs = TrainInputs(request)
    if not inputs.validate():
        errors = []
        for error in inputs.errors:
            errors.append({"message": error.message})
        return jsonify({"errors": errors}), 400

    try:
        # 解析传入的参数
        file_list = request.json["file_list"]
        space_name = request.json["space_name"]
        file_tag = request.json["file_tag"]
        index_path = "./training/models/{}.index".format(space_name)
        pkl_path = "./training/models/{}.pkl".format(space_name)

        trainingData = []

        # 根据传入的参数获取数据，如果获取数据的方式为 URL，则使用 requests 库发送 GET 请求获取数据
        if file_tag == "url":
            for url in file_list:
                r = requests.get(url)
                if r.status_code == 200:
                    # 如果训练数据是中文，则替换为英文
                    if "zh" in get_language_type(r.text):
                        en_content = translate(r.text)
                        trainingData.append(en_content)  # 将获取到的文本数据添加到 trainingData 列表中
                    else:
                        trainingData.append(r.text)
        else:
            # 如果获取数据方式为本地文件路径，则逐个读取文件并存储至 trainingData 列表中
            for file_path in file_list:
                with open(file_path, "r") as data_file:
                    use_text = data_file.read()
                trainingData.append(use_text)
                data_file.close()
        # 判断是否获取到训练数据，如果 trainingData 列表为空，则返回错误信息和 400 状态码
        if len(trainingData) < 1:
            return jsonify({"error": "No training data found"}), 400
        # 初始化文本拆分器和嵌入向量计算工具，并使用从文件中获取的训练数据创建向量存储器
        textSplitter = CharacterTextSplitter(chunk_size=2000, separator="\n")
        docs = []
        for sets in trainingData:
            docs.extend(textSplitter.split_text(sets))
        store = FAISS.from_texts(docs, OpenAIEmbeddings())
        faiss.write_index(store.index, index_path)
        store.index = None
        with open(pkl_path, "wb") as f:
            pickle.dump(store, f)

        # 如果需要下载 model 文件
        # return send_file("./training/models/{}.pkl".format(space_name), as_attachment=True)
        return (
            jsonify(
                {
                    "code": 0,
                    "msg": "success",
                    "data": {
                        "space_name": space_name,
                        "size_type": "kb",
                        "pkl_space_size": get_file_size(pkl_path),
                        "index_space_size": get_file_size(index_path),
                    },
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


class PromptUploadInput(Inputs):
    """
    针对 '/upload_prompt' 视图函数的输入参数规则
    """

    json = [
        JsonSchema(
            {
                "file_list": List[str],  # txt_url_list
            }
        )
    ]


@app.route("/upload_prompt", methods=["POST"])
def upload_prompt():
    # 使用我们定义的 PromptUploadInput 类进行参数验证，如果有错误则返回 400 状态码及错误信息
    inputs = PromptUploadInput(request)
    if not inputs.validate():
        errors = []
        for error in inputs.errors:
            errors.append({"message": error.message})
        return jsonify({"errors": errors}), 400

    try:
        # 解析传入的参数
        file_list = request.json["file_list"]

        all_file_size = 0
        for url in file_list:
            if url[-4:] != ".txt":
                return jsonify({"code": 1, "msg": "file is not txt"}), 400
            r = requests.get(url)
            if r.status_code == 200:
                file_save_path = "./training/prompt/{}".format(url.split("/")[-1])
                with open(file_save_path, "w", encoding="utf-8") as data_file:
                    data_file.write(r.text)
                    data_file.close()
                all_file_size += get_file_size(file_save_path)

        return (
            jsonify(
                {
                    "code": 0,
                    "msg": "success",
                    "data": {
                        "size_type": "kb",
                        "all_file_size": all_file_size,
                    },
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


class ChatInputs(Inputs):
    """
    针对 '/chat' 视图函数的输入参数规则
    """

    json = [
        JsonSchema(
            {
                "space_name": Union[str, int],
                "question": str,
                "history": List[Union[str, None]],
                "model_name": str,
                "prompt_name": Union[str, None],
            }
        )
    ]


@app.route("/chat", methods=["POST"])
def chat():
    inputs = ChatInputs(request)
    if not inputs.validate():
        errors = []
        for error in inputs.errors:
            errors.append({"message": error.message})
        return jsonify({"errors": errors}), 400
    try:
        # 获取请求参数
        space_name = request.json["space_name"]
        question = request.json["question"]
        history = request.json["history"]
        model_name = request.json["model_name"]
        prompt_name = request.json["prompt_name"] if "prompt_name" in request.json.keys() else "master"

        # 从文件中读取搜索引擎、向量空间等信息
        index = faiss.read_index("./training/models/{}.index".format(space_name))
        with open("./training/models/{}.pkl".format(space_name), "rb") as f:
            store = pickle.load(f)
        store.index = index

        with open("training/prompt/{}.txt".format(prompt_name), "r") as f:
            promptTemplate = f.read()
            f.close()

        # 根据 prompt 模板、历史文本、问题，构建预测使用的 LLM 模型
        prompt = Prompt(
            template=promptTemplate, input_variables=["history", "context", "question"]
        )

        llmChain = LLMChain(
            prompt=prompt,
            llm=OpenAI(
                temperature=0.25,
                model_name=model_name,
                openai_api_key=os.environ["OPENAI_API_KEY"],
            ),
        )

        # 通过搜索引擎获取文本上下文信息，结合模型产生回答
        docs = store.similarity_search(question)
        contexts = []
        for i, doc in enumerate(docs):
            contexts.append(f"Context {i}:\n{doc.page_content}")
        answer = llmChain.predict(
            question=question, context="\n\n".join(contexts), history=history
        )
        response = {
            "answer": answer,
            "history": history + [f"Human: {question}", f"Bot: {answer}"],
        }
        return jsonify({"code": 0, "msg": "success", "data": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

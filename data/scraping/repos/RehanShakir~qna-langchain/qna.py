from llama_index import download_loader, GPTVectorStoreIndex
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.utilities import ApifyWrapper
from langchain.indexes import VectorstoreIndexCreator
from langchain.docstore.document import Document
from langchain.retrievers import WikipediaRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import os


app = Flask(__name__)

load_dotenv()

CORS(app)

openai_api_key = os.getenv('OPENAI_API_KEY')


BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")

loader = BeautifulSoupWebReader()


@app.route('/generate-answer', methods=['POST'])
def ask_question():
    startAPICall = time.time()
    data = request.get_json()
    question = data.get("question")
    urls = data.get("urls")
    text = data.get("innerText")

    documents2 = loader.load_data(
        urls=urls)
    documents2[0].text = text

    index = GPTVectorStoreIndex.from_documents(documents2)

    query_engine = index.as_query_engine()
    tools = [
        Tool(
            name="Website Index",
            func=lambda q: query_engine.query(q),
            description=f"Provide answer questions about the text on websites. Provide detailed answers.",
        ),
    ]

    llm = OpenAI(temperature=0)

    memory = ConversationBufferMemory(memory_key="chat_history")
    agent_chain = initialize_agent(
        # zero-shot-react-description: This agent uses the ReAct framework to determine which tool to use based solely on the tool's description.
        # Any number of tools can be provided. This agent requires that a description is provided for each tool.
        tools, llm, agent="zero-shot-react-description", memory=memory,
    )
    output = agent_chain.run(input=question)

    endAPICall = time.time()
    diff = endAPICall - startAPICall
    print("diff", diff)
    if output:
        return jsonify({"output": output, "responseTime": diff})
    else:
        return jsonify({"error": "Missing 'question' parameter"}), 400


os.environ["APIFY_API_TOKEN"] = os.getenv('APIFY_API_TOKEN')

apify = ApifyWrapper()


@app.route('/apify-qna', methods=['POST'])
def ask_question_2():
    print("inn")
    loader = apify.call_actor(
        actor_id="apify/website-content-crawler",
        run_input={"startUrls": [
            {"url": "https://python.langchain.com/docs/integrations/chat/"}]},
        dataset_mapping_function=lambda item: Document(
            page_content=item["text"] or "", metadata={"source": item["url"]}
        ),
    )
    index = VectorstoreIndexCreator().from_loaders([loader])
    query = "Are any OpenAI chat models integrated in LangChain?"
    result = index.query(query)

    return jsonify({"output": result})


retriever = WikipediaRetriever()

model = ChatOpenAI(model_name="gpt-4")
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)


@app.route('/ask-wiki', methods=['POST'])
def ask_wiki():
    data = request.get_json()
    question = data.get("question")

    if question:
        result = qa({"question": question, "chat_history": []})
        answer = result['answer']
        return jsonify({"answer": answer})
    else:
        return jsonify({"error": "Missing 'question' parameter"}), 400


if __name__ == '__main__':
    app.run(debug=True)

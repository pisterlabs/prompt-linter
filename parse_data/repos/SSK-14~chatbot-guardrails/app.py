import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from nemoguardrails import LLMRails, RailsConfig
import pickle
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

with open("vectorstore/index.pkl", 'rb') as f: 
    vectorstore = pickle.load(f)

def get_qa_chain(llm, vectorstore):
    chain_type_kwargs = {"prompt": PROMPT}
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(), chain_type_kwargs=chain_type_kwargs, verbose=True)
    return qa_chain

def predict(message, history):
    config = RailsConfig.from_path("config")
    app = LLMRails(config)
    qa_chain = get_qa_chain(app.llm, vectorstore)
    app.register_action(qa_chain, name="qa_chain")
    ai_response = app.generate(message)
    return ai_response

demo = gr.ChatInterface(predict, chatbot=gr.Chatbot(height=500), examples=["How to host my website in aws?", "Tell me about AWS CodeStar?"], 
                        textbox=gr.Textbox(placeholder="Ask any question related to AWS", container=False, scale=6), title="AWS Bot", 
                        description="Experiment on langchain with NeMo Guardrails", theme="soft",)

if __name__ == "__main__":
    demo.launch()
import gradio as gr
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
import os
from langchain import HuggingFacePipeline

os.environ["HUGGINGFACEHUB_API_TOKEN"] = 'HUGGINGFACEHUB_API_TOKEN'
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")


class LLM:

    def file_upload(self, file):
        # 在这里添加文件上传的代码
        loader = TextLoader(file.name, encoding='utf8')
        documents = loader.load()
        # Text Splitters
        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # select embeddings
        embeddings = HuggingFaceEmbeddings()

        # create vectorstores
        db = Chroma.from_documents(texts, embeddings)

        # Retriever
        self.retriever = db.as_retriever(search_kwargs={"k": 2})

        return "file upload ok"

    def chatbot(self, text):
        # 在这里添加聊天机器人的代码，使用输入的文本作为聊天机器人的输入，并返回答复文本
        query = text
        docs = self.retriever.get_relevant_documents(query)
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0,
            top_p=0.,
            repetition_penalty=1.15
        )
        llm = HuggingFacePipeline(pipeline=pipe)

        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        return response


myLLM = LLM()

with gr.Blocks() as demo:
    with gr.Column():
        with gr.Column():
            result = gr.Interface(fn=myLLM.file_upload, inputs="file", outputs="text")

        with gr.Row():
            r = gr.Interface(fn=myLLM.chatbot, inputs="text", outputs="text")

demo.launch(share=True)
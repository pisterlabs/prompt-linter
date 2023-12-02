import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
import torch


class llmwebui:

    def file_upload(self, file):
        # 在这里添加文件上传的代码
        '''
        reader = PdfReader(file.name)
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
        '''
        with open(file.name, "r") as f:
            raw_text = f.read()
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        # 文本分割
        texts = text_splitter.split_text(raw_text)
        # 构建embedding
        embeddings = HuggingFaceEmbeddings(model_name='/Users/yingying/Desktop/pre_train_model/GanymedeNil_text2vec_large_chinese')

        # 搜索相近的embedding
        self.docsearch = FAISS.from_texts(texts, embeddings)
        self.step = 0
        self.chat_history_ids = None

        return "file upload ok"

    def chatbot(self, text):
        # 在这里添加聊天机器人的代码，使用输入的文本作为聊天机器人的输入，并返回答复文本
        query = text
        docs = self.docsearch.similarity_search(query)
        prompt = ""
        # 构建prompt
        for doc in docs:
            prompt = prompt + "\n" + doc.page_content
        prompt = prompt + "\nAnswer the question based on known information:{}".format(query)
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        # bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids],
        #                           dim=-1) if self.step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens,
        self.chat_history_ids = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        self.step += 1

        return tokenizer.decode(self.chat_history_ids[0], skip_special_tokens=True)


myllmwebui = llmwebui()

with gr.Blocks() as demo:
    with gr.Column():
        with gr.Column():
            result = gr.Interface(fn=myllmwebui.file_upload, inputs="file", outputs="text")

        with gr.Row():
            r = gr.Interface(fn=myllmwebui.chatbot, inputs="text", outputs="text")

demo.launch()
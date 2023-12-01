import wx
import requests
import threading
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os


class QAGUI(wx.Frame):
    def __init__(self, parent, title):
        super(QAGUI, self).__init__(parent, title=title, size=(700, 700))

        self.panel = wx.Panel(self)

        self.question_label = wx.StaticText(self.panel, label="请输入您的问题：")
        self.question_textbox = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE, size=(500, 30))
        self.submit_button = wx.Button(self.panel, label="提交问题")
        # 创建选择框控件
        self.answer_option_choices = ["直接生成答案", "接入向量数据库获取答案"]
        self.answer_option_choice = wx.Choice(self.panel, choices=self.answer_option_choices)
        self.answer_option_choice.SetSelection(1)  # 默认选中第一个选项

        self.dialogue_textbox = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE|wx.TE_READONLY, size=(500, 300))
        self.clear_dialogue_button = wx.Button(self.panel, label="清除对话记录")

        self.Bind(wx.EVT_BUTTON, self.on_submit, self.submit_button)
        self.Bind(wx.EVT_BUTTON, self.clear_dialogue, self.clear_dialogue_button)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.question_label, 0, wx.ALL, 10)
        self.sizer.Add(self.question_textbox, 0, wx.ALL, 10)
        self.sizer.Add(self.submit_button, 0, wx.ALL, 10)
        self.sizer.Add(self.answer_option_choice, 0, wx.ALL, 10)
        self.sizer.Add(self.dialogue_textbox, 0, wx.ALL, 10)
        self.sizer.Add(self.clear_dialogue_button, 0, wx.ALL, 10)

        self.panel.SetSizer(self.sizer)
        self.Show()
        
        threading.Thread(target=self.init_arxiv).start()
        
    def init_arxiv(self):
        os.environ["OPENAI_API_KEY"] = "None"
        os.environ["OPENAI_API_BASE"] = "http://172.29.7.155:8000/v1"
        llm = ChatOpenAI(model_name="vicuna-13b-v1.5", temperature=0.3, max_tokens=256)
        prompt = PromptTemplate(
            input_variables=["docs", "question"],
            template="""Hello! Now I'll give you some docs, please read the papers: {docs}. Please pay attention to the access id of the papers since you will be required to provide them later. After Reading the papers, I would like you to answer this question arrcording to the papers: {question}, please note that all the points in your answer should be surpported by the papers I give you and you are required to list title and authors of the papers that surpports your points. You are also required to give the paper's access address like https://arxiv.org/abs/access_id  . Finally if you really don't know how to answer, you can truthfully say you don't know.
            This is a brief example:
            input": "What is natural language processing?
            "output": "Natural language processing (NLP) is a branch of artificial intelligence (AI) that focuses on the interaction between computers and human languages. \  It involves the use of computational techniques to process, analyze, and generate natural language text and speech. \  One reference paper that provides a comprehensive survey of NLP is 'Natural Language Processing - A Survey' by Y. Liu et al. (2012), which can be accessed at <https://arxiv.org/abs/1209.6238>;. \  Another relevant paper is 'Parsing of part-of-speech tagged Assamese Texts' by B. K. Deka et al. (2009), which can be accessed at <https://arxiv.org/abs/0912.1820>;. \  For a more detailed understanding of NLP, 'Du TAL au TIL' by D. Sauperl (2012) is a useful resource, available at <https://arxiv.org/abs/1201.4733>;.
            """,
        )
        self.chain = LLMChain(llm=llm, prompt=prompt)
        embedding = HuggingFaceEmbeddings() 
        self.db = Milvus(embedding_function=embedding, collection_name="arXiv_prompt",connection_args={"host": "172.29.4.47", "port": "19530"})

    def on_submit(self, event):
        # 在这里处理用户提交问题的逻辑，根据复选框的状态生成答案或对话
        question = self.question_textbox.GetLineText(0)
        # 在单独的线程中进行网络请求
        # 在聊天记录框中添加等待提示
        self.dialogue_textbox.AppendText("用户: \n" + question + '\n')
        self.dialogue_textbox.AppendText("系统: \n" + "正在生成答案，请稍候...\n")
        self.choice = self.answer_option_choices[self.answer_option_choice.GetSelection()]
        threading.Thread(target=self.fetch_answer, args=(question,self.choice)).start()

    def fetch_answer(self, question, selected_option):
        
        ans = ""

        if selected_option == "接入向量数据库获取答案":
            docs = self.db.similarity_search(question)
            # print(docs)
            ans = self.chain.run({
                'docs': docs,
                'question': question
            })
        else: 
            headers = {"Content-Type": "application/json"}
            parms = {
                "model": "vicuna-13b-v1.5",
                "prompt": question,
                "temperature": 0.7,
                "max_tokens": 256,
                "stop": None,
                "n": 1,
                "top_p": 1.0,
            }
            response = requests.post("http://172.29.7.155:8000/v1/completions", headers=headers, json=parms)
            ans = response.json()["choices"][0]["text"]

        # 在主线程中更新 GUI
        wx.CallAfter(self.update_dialogue, question, ans)

    def update_dialogue(self, question, ans):
        # 替换等待提示为真正的回答
        self.dialogue_textbox.Remove(self.dialogue_textbox.GetLastPosition() - 14, self.dialogue_textbox.GetLastPosition())
        self.dialogue_textbox.AppendText(ans + '\n')

    # ... (其他方法)
    def clear_dialogue(self, event):
        self.dialogue_textbox.Clear()

if __name__ == '__main__':
    app = wx.App()
    QAGUI(None, title="语言模型问答系统")
    app.MainLoop()

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
from langchain.prompts import (ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate)
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import (get_openai_callback)


# LLMs（大语言模型）- 简单使用LLM模型进行对话
def generate_llm_reply():
    message = "给我讲一个笑话"
    llm = OpenAI(model_name="text-davinci-003", n=2, temperature=0.3)
    msg = llm(message)
    print(msg)


# 聊天模型 - 使用Chat模型进行对话
def generate_chat_reply():
    messages = [SystemMessage(content="返回json object，不要纯文本，按照每项参数拆分，不要说明和解释信息"), HumanMessage(content="告诉我model Y汽车的尺寸参数")]
    chat = ChatOpenAI(temperature=0)
    print(chat(messages))


# 提示模板 - 使用Template进行对话
def generate_template_reply():
    system_template = "你是一个把{input_language}翻译成{output_language}的助手"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    messages = chat_prompt.format_prompt(input_language="英语", output_language="汉语", text="I love programming.")
    print(messages)  # messages=[SystemMessage(content='你是一个把英语翻译成汉语的助手', additional_kwargs={}), HumanMessage(content='I love programming.', additional_kwargs={}, example=False)]

    chat = ChatOpenAI(temperature=0)
    print(chat(messages.to_messages()))  # content='我喜欢编程。' additional_kwargs={} example=False


# 文本嵌入模型（Text Embedding Model） - 使用Text Embedding模型进行对话
def generate_text_embedding_reply():
    embeddings = OpenAIEmbeddings()
    text = "这是一个测试文档。"

    # 在这段代码中我们使用了embed_query和embed_documents两个方法，它们最大的不同是embed_query接收一个字符串作为输入，而embed_documents可以接收一组字符串，一些模型自身划分了这两个方法，LangChain也保留了下来。
    query_result = embeddings.embed_query(text)
    doc_result = embeddings.embed_documents([text])

    print(query_result)
    print(doc_result)


# 获取Token消耗量
def get_token_consumption():
    llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)
    with get_openai_callback() as cb:
        result = llm("Tell me a joke")
        print(cb)
        print(result)
        # Tokens Used: 42
        #         Prompt Tokens: 4
        #         Completion Tokens: 38
        # Successful Requests: 1
        # Total Cost (USD): $0.00084


if __name__ == '__main__':
    # generate_llm_reply()
    # generate_chat_reply()
    # generate_template_reply()
    # generate_text_embedding_reply()
    get_token_consumption()

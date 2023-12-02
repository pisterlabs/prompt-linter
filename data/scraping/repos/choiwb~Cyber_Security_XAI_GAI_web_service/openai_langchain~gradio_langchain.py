import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
import os
import re
import time


# (회사) 유료 API 키!!!!!!!!
# 20230904_AIR	
os.environ['OPENAI_API_KEY'] = "YOUR OPENAI API KEY !!!!!!!"
    
raw_data_path = "YOUR DATA PATH !!!!!!!"

# loader = TextLoader(os.path.join(raw_data_path, 'mitre_attack_20230823.txt'))
# documents = loader.load()

# text_splitter = CharacterTextSplitter(        
#     # pdf 전처리가 \n\n 으로 구성됨
#     separator = "\n\n",
#     chunk_size = 3200,
#     chunk_overlap  = 0,
#     length_function = len,
# )

# texts = text_splitter.split_documents(documents)
# print(len(texts))



# OpenAI VS HuggingFace
# embeddings = OpenAIEmbeddings()
# HuggingFace의 경우, 서버에 gpu가 있어야 빠를 것으로 보임.!!!!!!!!!!!!!!
embeddings = HuggingFaceEmbeddings()


callbacks = [StreamingStdOutCallbackHandler()]

llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0, max_tokens=512,
                 callbacks=callbacks, streaming=True)
# llm = OpenAI(model_name='gpt-3.5-turbo-instruct', temperature=0, max_tokens=512,
#                    callbacks=callbacks, streaming=True)

template = """You are a cyber security analyst. about user question, answering specifically in korean.
            Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            For questions, related to Mitre Att&ck, in the case of the relationship between Tactics ID and T-ID (Techniques ID), please find T-ID (Techniques ID) based on Tactics ID.
            Tactics ID's like start 'TA' before 4 number.
            T-ID (Techniques ID) like start 'T' before 4 number.
            Tactics ID is a major category of T-ID (Techniques ID), and has an n to n relationship.
            In particular Enterprise Tactics ID consist of 1TA0001 (Initial Access), TA0002 (Execution), TA0003 (Persistence), 
            TA0004 (Privilege Escalation), TA0005 (Defense Evasion), TA0006 (Credential Access), TA0007 (Discovery), 
            TA0008 (Lateral Movement), TA0009 (Collection), TA0010 (Exfiltration), TA0011 (Command and Control),
            TA0040 (Impact), TA0042 (Resource Development), TA0043 (Reconnaissance).
            Respond don't know to questions not related to cyber security.
            Use three sentences maximum and keep the answer as concise as possible. 
            {context}
            question: {question}
            answer: """


QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)


'''Mitre Att&ck의 공식 홈페이지를 통해 검색한 결과를 임베딩 db에서 제일 유사도가 높은 질문의 답변과 비교하여 공식 홈페이지 기준 출력 logic 구현 !!!!!!!!'''


################################################################################
# 임베딩 벡터 DB 저장 & 호출
db_save_path = "DB SAVE PATH !!!!!!!"

# start = time.time()
# docsearch = FAISS.from_documents(texts, embeddings)
# docsearch.embedding_function
# docsearch.save_local(os.path.join(db_save_path, "mitre_attack_20230908_index"))
# end = time.time()
# print('임베딩 완료 시간: %.2f (초)' %(end-start))

new_docsearch = FAISS.load_local(os.path.join(db_save_path, 'mitre_attack_20230908_index'), embeddings)

retriever = new_docsearch.as_retriever(search_type="similarity", search_kwargs={"k":1})

# # 유사도 0.7 이상만 추출
embeddings_filter = EmbeddingsFilter(embeddings = embeddings, similarity_threshold = 0.7)

# # 압축 검색기 생성
compression_retriever = ContextualCompressionRetriever(base_compressor = embeddings_filter,
                                                        base_retriever = retriever)
################################################################################


retrieval_qa_chain = RetrievalQA.from_chain_type(llm,
                                        retriever=compression_retriever, 
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                                        chain_type='stuff'
                                        )
qa_llmchain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)


conversation_history = []

def query_chain(question):
    
    # 질문을 대화 기록에 추가
    conversation_history.append(("latest question: ", question))

    # 대화 맥락 형식화: 가장 최근의 대화만 latest question, latest answer로 나머지는 priorr question, prior answer로 표시
    if len(conversation_history) == 1:
        # print('대화 시작 !!!!!!!')
        formatted_conversation_history = f"latest question: {question}"
    else:
        formatted_conversation_history = "\n".join([f"prior answer: {text}" if sender == "latest answer: " else f"prior question: {text}" for sender, text in conversation_history])
        
        # formatted_conversation_history의 마지막 prior question은 아래 코드 에서 정의한 latest question과 동일하므로 일단 제거 필요
        lines = formatted_conversation_history.split('\n')
        if lines[-1].startswith("prior question:"):
            lines.pop()
        formatted_conversation_history = '\n'.join(lines)
        
        formatted_conversation_history += f"\nlatest question: {question}"
    # print('전체 대화 맥락 기반 질문: ', formatted_conversation_history)

    # print(11111111111111111111111111111111111111111111111111111111111)
    result = retrieval_qa_chain({"query": formatted_conversation_history}) 
    # context = retrieval_qa_chain({"query": formatted_conversation_history})

    # print(22222222222222222222222222222222222222222222222222222222222)
    # result = qa_llmchain({"context": context, "question": formatted_conversation_history})
        
    print('답변에 대한 참조 문서')
    print(result['source_documents'])    
    
    # 답변을 대화 기록에 추가 => 추 후, AIR 적용 시, DB 화 필요 함!!!!!
    conversation_history.append(("latest answer: ", result["result"]))

    return result["result"]
    # return result



def generate_text(history):
    generated_history = history.copy()

    stop_re = re.compile(r'(question:)', re.MULTILINE)
 
    # respomse는 최신 답변만 해당 !!!!!!!!!
    response = query_chain(generated_history[-1][0])  # Assuming the user message is the last one in history    
    
    if re.findall(stop_re, response):
        response = ''.join(response.split('\n')[0])

    history[-1][1] = ""
    for character in response:
        generated_history[-1][1] += str(character)
        time.sleep(0.03)
        yield generated_history

            
with gr.Blocks(title= 'IGLOO AiR ChatBot', css="#chatbot .overflow-y-auto{height:5000px} footer {visibility: hidden;}") as gradio_interface:

    with gr.Row():
        gr.HTML(
        """<div style="text-align: center; max-width: 2000px; margin: 0 auto; max-height: 5000px; overflow-y: hidden;">
            <div>
                <h1>IGLOO AiR ChatBot</h1>
            </div>
        </div>"""

        )

    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot()
            # msg = gr.Textbox(value="SQL Injection 공격에 대응하는 방법을 알려주세요.", placeholder="질문을 입력해주세요.")
            msg = gr.Textbox(value="Mitre Att&ck에 대해서 설명해주세요.", placeholder="질문을 입력해주세요.")

            with gr.Row():
                clear = gr.Button("Clear")



    def user(user_message, history):
        # user_message 에 \n, \r, \t, "가 있는 경우, ' ' 처리
        user_message = user_message.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('"', ' ')

        return "", history + [[user_message, None]]
    
    def fix_history(history):
        update_history = False
        for i, (user, bot) in enumerate(history):
            if bot is None:
                update_history = True
                history[i][1] = "_silence_"
        if update_history:
            chatbot.update(history) 

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=True).then(
        # generate_text 함수의 경우, 대화의 history 를 나타냄.
        generate_text, inputs=[
            chatbot
        ], outputs=[chatbot],
    ).then(fix_history, chatbot)

    clear.click(lambda: None, None, chatbot, queue=True)

gradio_interface.queue().launch(debug=True, server_name="127.0.0.1", share=True)


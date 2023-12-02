from dotenv import load_dotenv
import os

load_dotenv()

import openai
import torch
from sentence_transformers import SentenceTransformer, util
import socket
from _thread import *
from transformers import pipeline
from papago_long import translate
import chatGPT_api
import datetime
from Googlecalendar import add_calendar
from chatGPT_api import chatGPT

openai.organization = os.environ.get('SEONGWAN_organization')
openai.api_key = os.environ.get('SEONGWAN_YOUR_API_KEY')
YOUR_API_KEY = os.environ.get('YOUR_API_KEY') # chatGPT API KEY
openai.Engine.list()

ButtonPression = 0 # 0 : 안눌렸을때, 1 : 발표 시작, 2 : 발표 끝

# Text embedding model
MODEL = "text-embedding-ada-002"
# 우리의 AI 비서가 처리 가능한 명령어
COMMAND = ["A가 한 말 요약,정리해줘", "지금까지 회의 내용 요약,정리해줘" ,"캘린더에 저장해줘", "회의 참여도 알려줘"]
# Socket connection parameters
HOST = '192.168.204.230'
PORT = 9999

speaker_word_count = dict()
client_sockets = []
whole_transcript = []
button_transcript = []
client_transcript = {}

def max_similaritys_command(query) : # 사용자의 입력값 중에서 가장 우리 command와 유사한거 가져오기
    max_similaritys = []
    query_embedding = openai.Embedding.create(input=[query], engine=MODEL)
    query_tensor = torch.tensor(query_embedding['data'][0]['embedding'])
    # document embedding
    document_embedding = openai.Embedding.create(input=COMMAND, engine=MODEL)
    for i,document in enumerate(document_embedding['data']) :
        document_tensor = torch.tensor(document_embedding['data'][i]['embedding'])
        similarity = (float(util.cos_sim(query_tensor, document_tensor)),COMMAND[i])
        max_similaritys.append(similarity)
    max_similaritys.sort(key= lambda x:-x[0])

    user_command = max_similaritys[0][1] # 유저가 입력한 커멘드 중에서 우리의 커멘드와 가장 유사한거 일치시키기

    ### 사용자 한명 실시간 발표 요약
    if user_command == "A가 한 말 요약,정리해줘" or ButtonPression == 2: 
        # 발표 버튼 누르면 서버로 알려줘 #     
        name, script = get_user_script(query) # 사용자의 입력값에서 명령어랑 타겟 이름 꺼내오기
        if(ButtonPression == 2) : # 발표 종료 버튼이 눌렸을 때
            script = get_full_presentation() # 발표 시작부터 끝까지 텍스트 저장
            button_transcript = []
        eng_script = translate("krTOen",script) # papago로 영어로 번역
        eng_summerize = summerize_model(eng_script) # 영어로 번역한 발표 요약
        kor_summerize = translate("enTOkr",eng_summerize) # 다시 한국어로 번역
        result = name + "이 한 말을 요약해봤어요 :)\n" + kor_summerize
        return result

    ### 지금까지 회의 내용 요약
    elif user_command == "지금까지 회의 내용 요약,정리해줘" : 
        script = get_full_script() # 모든 회의록 가져오기
        kor_summerize = translate("enTOkr",summerize_model(translate("krTOen",script)))
        result = "지금까지의 회의 내용을 요약해 보았아요 :)\n" + kor_summerize
        return result

    ### 회의에서 나온 요일 캘린더에 저장
    elif user_command == "캘린더에 저장해줘" : 
        # command를 가장 최근에 한 말로 넣기
        for key, value in whole_transcript[-1].items() : # 가장 최근에 한 말 input으로 넣기
            script = value 
        date_string = chatGPT_api.chatGPT(user_command, script, YOUR_API_KEY)
        date_time_string = date_string.split("\n")[0].split(":")[1].strip()
        topic = date_string.split("\n")[1].split(":")[1].strip()
        date_time_obj = datetime.datetime.strptime(date_time_string, "%m월 %d일 오후 %H시")
        add_calendar(date_time_obj, topic)
        return "구글 캘린더에 일정을 저장 완료했어요 :)"

    ### 회의 참여도 알려주기
    elif user_command == "회의 참여도 알려줘" :
        result = ""
        result += "발화 기반 회의 참여도 순위는 다음과 같습니다.\n"
        participation = sort_transcript_by_length(get_speaker_word_count(whole_transcript))
        for speaker in participation :
            result += (speaker + "\n")
        return result
    ### chatGPT를 통한 질의응답
    else :
        english_full_script = translate("krTOen",get_full_script())
        english_command = translate("krTOen",query)
        QNA_scipt = chatGPT(english_command, english_full_script)
        return QNA_scipt
        # chatGPT에서 에러가 발생 했을 경우
        if max_similaritys[-1][0] < 0.7 : # 이상한 명령어가 들어 왔을 경우
            return "chatty cat이 이해할수 없는 명령어에요😢 다른 명령어를 입력해주세요"

def summerize_model(data) : # 요약 모델
    summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY")
    return summarizer(data)

def get_username(command) : # command에서 유저 이름 빼오기
    names = set()
    # script에서 이름 꺼내오기
    for name in whole_transcript :
        for key in name.keys() :
            names.add(key)
    # 명령어에서 이름 꺼내오기
    for name in names :
        if name in command :
            return name

def get_user_script(command) : # 특정 유저가 지금까지 한 말 요약하기
    name = get_username(command)
    text = ""
    for transcript in whole_transcript :
        for key,value in transcript.items() :
            if(key == name) :
                text += value
    return (name, text)

def get_speaker_word_count(transcript): # 사람별로 말한 문자열 길이 딕셔너리 반환하는 함수
    speaker_word_count = {}
    for item in transcript:
        for speaker, speech in item.items():
            if speaker in speaker_word_count:
                speaker_word_count[speaker] += len(speech)
            else:
                speaker_word_count[speaker] = len(speech)
    return speaker_word_count

def sort_transcript_by_length(transcript_dict): # 딕셔너리의 value 값 기준으로 정렬
    return dict(sorted(transcript_dict.items(), key=lambda item: item[1], reverse=True))

def get_full_script() : # 지금까지 회의 내용 요약하기
    text = ""
    for transcript in whole_transcript :
        for key,value in transcript.items() :
                text += value
    return text

def get_full_presentation() : # 지금까지 회의 내용 요약하기
    text = ""
    for transcript in button_transcript :
        for key,value in transcript.items() :
                text += value
    return text

# command에서 openAI text similarity로 우리 명령어 찾기
# commamd에서 이름 뽑아서 script에서 찾기

# Dedicated thread function for receiving speech text from each users
def threaded(client_socket, addr):
    print('>> Connected by :', addr[0], ':', addr[1])

    # Repeat until user disconnects
    while True:
        try:
            data = client_socket.recv(1024)
            if not data:
                print('>> Disconnected by ' + addr[0], ':', addr[1])
                break
            text_data, username = data.decode().split(';')
            print('>> Received from : ' + username," data : ", text_data)
            whole_transcript.append({username:text_data})
            if(ButtonPression == 1) : # 발표시작 버튼이 눌렸을 때
                button_transcript.append({username:text_data})
            speaker_word_count = sort_transcript_by_length(get_speaker_word_count(whole_transcript))
            """
            for key,value in whole_transcript :
                print("name : ",key)
                print("data : ",value)
            """
        except ConnectionResetError as e:
            print('>> Disconnected by ' + addr[0], ':', addr[1])
            break 

    if client_socket in client_sockets :
        client_sockets.remove(client_socket)
        print('remove client list : ',len(client_sockets))

    client_socket.close()

def main():
    print('>> Server Start')
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()

    try:
        while True:
            print('>> Waiting for connection...')
            client_socket, addr = server_socket.accept()
            client_sockets.append(client_socket)
            client_transcript[addr[0]] = [] # addr[0] = 192.168.1.10
            start_new_thread(threaded, (client_socket, addr))
            print("Number of clients: ", len(client_sockets))
            
    except Exception as e :
        print (e)
    finally:
        server_socket.close()

if __name__ == "__main__":
    main()
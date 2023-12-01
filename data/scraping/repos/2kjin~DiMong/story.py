import os
from dotenv import load_dotenv
import sys
import csv
import openai
import time
import tts

dinos = ["티라노사우루스가",
"트리케라톱스가",
"벨로시랩터가",
"파라사우롤로푸스가",
"이구아노돈이",
"프테라노돈이",
"안킬로사우루스가",
"아르헨티노사우루스가",
"파키케팔로사우루스가",
"엘라스모사우루스가",
"모사사우루스가",
"오우라노사우루스가",
"인시시보사우루스가",
"람베오사우루스가",
"노도사우루스가",
"오비랍토르가",
"케찰코아틀루스가",
"켄트로사우루스가",
"스피노사우루스가",
"스테고사우르스가",
"브라키오사우루스가",
"알로사우루스가",
"딜로포사우루스가",
"다센트루루스가",
"친타오사우루스가",
"콤프소그나투스가",
"기가노토사우루스가", 
"디모르포돈이",
"사우로파가낙스가",
"카스모사우루스가"]

topics = [
    "아름다운","재밌는","흥미로운"
]
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if True :
    for t in range(0,3):
        topic = topics[t]
        i=1
        for dino in dinos :
            print(dino+" 주인공인 "+topic+" 동화를 아이들에게 짧게 들려줘 ")
    pass
if False :
    for t in range(0,3):
        topic = topics[t]
        i=1
        for dino in dinos :
            stime = time.time()
            print(str(i)+':start')
            i+=1
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content":"You are smart storyteller"},
                {"role": "user", "content": dino+" 주인공인 "+topic+" 동화를 아이들에게 짧게 들려줘 "},
            ],
            )
            etime = time.time()
            print("time:",etime-stime)
            tts.tts(completion["choices"][0]["message"]["content"],dino[:-1])
if False : 
    completion = {
    "choices": [
        {
        "finish_reason": "stop",
        "index": 0,
        "message": {
            "content": "\uc61b\ub0a0 \uc61b\uc801\uc5d0, \ud2f0\ub77c\ub178\uc0ac\uc6b0\ub974\uc2a4\ub4e4\uc740 \uc9c0\uad6c\uc0c1\uc5d0\uc11c \uac00\uc7a5 \uac15\ud55c \ub3d9\ubb3c\uc774\uc5c8\uc2b5\ub2c8\ub2e4. \uadf8\ub4e4\uc740 \ub9e4\uc6b0 \ud06c\uace0 \uac15\ub825\ud574\uc11c, \ub2e4\ub978 \ub3d9\ubb3c\ub4e4\uc774 \uadf8\ub4e4\uc744 \ucad3\uc544\ub2e4\ub2c8\uc9c0 \uc54a\uc558\uc2b5\ub2c8\ub2e4.\n\n\ud558\uc9c0\ub9cc, \ud55c \ubc88\uc740 \uc791\uc740 \ub3d9\ubb3c\ub4e4\uc774 \ud2f0\ub77c\ub178\uc0ac\uc6b0\ub974\uc2a4\ub97c \uc774\uae30\ub294 \ubc29\ubc95\uc744 \ubc1c\uacac\ud588\uc2b5\ub2c8\ub2e4. \uadf8\ub4e4\uc740 \uaf2c\ub9ac\ub97c \ubb3c\uc5b4\uc11c, \ud2f0\ub77c\ub178\uc0ac\uc6b0\ub974\uc2a4\ub97c \uc4f0\ub7ec\ub728\ub9ac\uae30 \uc2dc\uc791\ud588\uc2b5\ub2c8\ub2e4. \uc774\ud6c4\ub85c, \ud2f0\ub77c\ub178\uc0ac\uc6b0\ub974\uc2a4\ub4e4\uc740 \ub354 \uc774\uc0c1 \ub2e4\ub978 \ub3d9\ubb3c\ub4e4\uc744 \ubb34\uc2dc\ud560 \uc218 \uc5c6\uac8c \ub418\uc5c8\uc2b5\ub2c8\ub2e4.\n\n\uadf8\ub807\uac8c \ud574\uc11c, \ud2f0\ub77c\ub178\uc0ac\uc6b0\ub974\uc2a4\ub4e4\uc740 \ub354 \uc774\uc0c1 \uc9c0\uad6c\uc0c1\uc5d0\uc11c \uac00\uc7a5 \uac15\ub825\ud55c \ub3d9\ubb3c\uc774 \uc544\ub2c8\uac8c \ub418\uc5c8\uc2b5\ub2c8\ub2e4. \uc774 \uc774\uc57c\uae30\ub97c \ub4e4\uc740 \uc544\uc774\ub4e4\uc740, \ud06c\uae30\uc640 \uac15\ub825\ud568\uc774 \uc911\uc694\ud55c \uac83\uc774 \uc544\ub2c8\ub77c, \uc9c0\ud61c\uc640 \ud611\ub825\uc774 \uc911\uc694\ud558\ub2e4\ub294 \uac83\uc744 \ubc30\uc6b8 \uc218 \uc788\uc744 \uac81\ub2c8\ub2e4.",
            "role": "assistant"
        }
        },
        {
        "finish_reason": "stop",
        "index": 0,
        "message": {
            "content": "\uc6b0\ub9ac\uac00 \uc54c\uace0 \uc788\ub294 \ub300\ubd80\ubd84\uc758 \uacf5\ub8e1\ub4e4\uc740 \uc5ec\ub7ec \uac00\uc9c0 \uc774\uc720\ub85c \uba78\uc885\ud574\ubc84\ub838\uc2b5\ub2c8\ub2e4. \ud558\uc9c0\ub9cc \ud2f0\ub77c\ub178\uc0ac\uc6b0\ub974\uc2a4\ub294 \uadfc\uc131\uc774 \uac15\ud558\uc5ec \ub9ce\uc740 \uc2dc\uac04\ub3d9\uc548 \uc0b4\uc544\ub0a8\uc558\uc2b5\ub2c8\ub2e4.\n\n\ud558\uc9c0\ub9cc \uadf8 \uc911\uc5d0\ub294 \ub9e4\uc6b0 \ud2b9\ubcc4\ud55c \ud2f0\ub77c\ub178\uc0ac\uc6b0\ub974\uc2a4\uac00 \uc788\uc5c8\uc2b5\ub2c8\ub2e4. \uadf8 \uc774\ub984\uc740 \"\uc2a4\ud2f0\ube44\" \uc785\ub2c8\ub2e4. \uc774 \ud2f0\ub77c\ub178\uc0ac\uc6b0\ub974\uc2a4\ub294 \ub2e4\ub978 \uacf5\ub8e1\ub4e4\uacfc\ub294 \uc870\uae08 \ub2e4\ub978 \ud2b9\uc9d5\uc744 \uac00\uc9c0\uace0 \uc788\uc5c8\ub294\ub370\uc694. \ubc14\ub85c \ub9e4\uc6b0 \uc791\uc740 \ud314\uacfc \uc190\uac00\ub77d\uc744 \uac00\uc9c0\uace0 \uc788\uc5c8\uc2b5\ub2c8\ub2e4.\n\n\uadf8\ub7ec\ub098 \uc791\uc740 \ud314\uacfc \uc190\uac00\ub77d\uc774 \uc2a4\ud2f0\ube44\uc5d0\uac8c \ud070 \ub3c4\uc6c0\uc774 \ub418\uc5c8\ub294\ub370\uc694. \uadf8\ub294 \ubb34\uc5c7\uc774\ub4e0 \uc798 \uc7a1\uc544\ub0bc \uc218 \uc788\uc5c8\uace0, \uc791\uc740 \ubc8c\ub808\ub3c4 \uc27d\uac8c \uc7a1\uc544\uba39\uc744 \uc218 \uc788\uc5c8\uc2b5\ub2c8\ub2e4.\n\n\uadf8\ub9ac\uace0 \uadf8\ub294 \ub9e4\uc6b0 \uce5c\uadfc\ud574\uc11c \ub2e4\ub978 \uacf5\ub8e1\ub4e4\uacfc \ud568\uaed8 \ub178\ub294 \uac83\uc744 \uc88b\uc544\ud588\uc2b5\ub2c8\ub2e4. \uc791\uc740 \uc190\uac00\ub77d\uc73c\ub85c \uacf5\uc744 \ub358\uc9c0\uac70\ub098, \uac19\uc774 \ub204\uc6cc \uc26c\uac70\ub098, \uba39\uac70\ub9ac\ub97c \ub098\ub204\uace4 \ud588\uc2b5\ub2c8\ub2e4.\n\n\uc2a4\ud2f0\ube44\ub294 \ub9ce\uc740 \uc0ac\ub78c\ub4e4\uc5d0\uac8c \uc0ac\ub791\ubc1b\uc558\uace0, \uadf8\uc758 \uc720\ubb3c\ub4e4\uc740 \ubc29\ub300\ud55c \ubc15\ubb3c\uad00\uc5d0 \uc804\uc2dc\ub418\uc5b4 \uc788\uc2b5\ub2c8\ub2e4. \uc774\ub7f0 \ud2b9\ubcc4\ud55c \ud2f0\ub77c\ub178\uc0ac\uc6b0\ub974\uc2a4\uc758 \uc774\uc57c\uae30\ub97c \uc544\uc774\ub4e4\uc5d0\uac8c \ub4e4\ub824\uc8fc\uba74, \uacf5\ub8e1\uc5d0 \ub300\ud55c \ub354 \ub9ce\uc740 \uad00\uc2ec\uacfc \uad00\uc2ec\uc744 \ub04c\uc5b4\ub0bc \uc218 \uc788\uc744 \uac83\uc785\ub2c8\ub2e4!",
            "role": "assistant"
        }
        }
    ],
    "created": 1680051499,
    "id": "chatcmpl-6zEQlXz9RwWKsmkuHfb0Jf60eFX2W",
    "model": "gpt-3.5-turbo-0301",
    "object": "chat.completion",
    "usage": {
        "completion_tokens": 306,
        "prompt_tokens": 42,
        "total_tokens": 348
    }
    }
    d = {
        "prop1" : "Hi"
    }
    print(d["prop1"])
    print(completion["choices"][0]["message"]["content"])
    print(completion["choices"][1]["message"]["content"])


# with open('dinodetail.csv', newline='',encoding='cp949') as csvfile:
#     details = csv.reader(csvfile)
#     #공룡 정보 반복문 - polly에 텍스트 입력하고 , 공룡이름 으로 된 음성파일 저장
#     for name,detail in details:
#         print("name:",name)
#         print("detail:",detail)
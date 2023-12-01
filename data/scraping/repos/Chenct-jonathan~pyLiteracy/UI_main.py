#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request
import json
import openai
import os
import sys
import re
from naive_typo import typoChecker
from ArticutAPI import Articut
sys.path.append('../../loki_models/Gua_Zai')
from Gua_Zai import execLoki

BASEPATH = os.path.dirname(os.path.abspath(__file__))
try:
    with open("{}/account.info".format(BASEPATH), encoding="utf-8") as f:
        accountDICT = json.load(f)
except:
    print("提示！！目前使用 Articut 每小時公用字數中！！")
    accountDICT = {"username":"", "apikey":""}
pat = re.compile("</?\w+?_?\w*?>")
articut = Articut(username=accountDICT["username"], apikey=accountDICT["apikey"])
try:
    openai.api_key = accountDICT["gaikey"]
except:
    openai.api_key = ""

app = Flask(__name__)

@app.route("/")
def home():
    #return render_template("homepage.html")
    return render_template("pyLiteracy.html")

@app.route("/gua", methods=["POST"])
def zaiChecker():
    if request.method == "POST":# or request.method == "GET":
        sentenceLIST = []
        inputDICT = request.json
        inputSTR = inputDICT["inputSTR"]          #此行對應 .js 中的：payload["inputSTR"] = $("#inputSTR").val(); //將輸入區內的字串值載入 payload 中，給定 key 為 "inputSTR"
        if inputSTR.strip() == "":                #檢查一下，如果送空白字串上來，就回覆空字串。
            return jsonify({"returnData":""})

        typoLIST = typoChecker(inputSTR)

        #<Loki 的計算區塊>
        articutDICT = articut.parse(inputSTR)     #如果不是空字串，就把字串送給 Articut 處理以便斷句。
        #zaiCount = 0
        if articutDICT["status"] == True:         #若斷句結果正常結束，就繼續往下走。否則就回覆 jsonify() 後的結果。
            pass
        else:
            return jsonify({"returnData": articutDICT["msg"]})

        for i in articutDICT["result_pos"]:       #將 Articut 處理後的每一句，送入 Loki 模型中處理。
            if len(i) <= 1:
                sentenceLIST.append(i)
            elif "<FUNC_inner>在</FUNC_inner>" in i or "<ASPECT>在</ASPECT>" in i:
                #for j in i:
                    #if j == "<FUNC_inner>在</FUNC_inner>" or j == "<ASPECT>在</ASPECT>":
                        #zaiCount += 1
                #if zaiCount > 1:
                    #i =  re.sub(r"<FUNC_inner>在</FUNC_inner><[^<]*?>[^<]*?</[^<]*?>", "", i)
                checkSTR = re.sub(pat, "", i)
                checkResultDICT = execLoki(checkSTR)
                if checkResultDICT["Zai"] != []:
                    sentenceLIST.append(checkSTR)
                else:
                    if "<FUNC_inner>在</FUNC_inner>" in i:
                        checkSTR = checkSTR.replace("在", "[btn]在[/btn]").replace("[btn]", "<button type='button' class='btn btn-danger danger-border' data-bs-toggle='tooltip' data-bs-placement='top' title='「再」啦！'>").replace("[/btn]", "</button>")#re.sub(pat, "", i.replace("<FUNC_inner>在</FUNC_inner>", "[btn]<FUNC_inner>在</FUNC_inner>[/btn]".format(i))).replace("[btn]", "<button type='button' class='btn btn-danger danger-border' data-bs-toggle='tooltip' data-bs-placement='top' title='「再」啦！'>").replace("[/btn]", "</button>")
                    else: #"<ASPECT>在</ASPECT>"
                        checkSTR = checkSTR.replace("在", "[btn]在[/btn]").replace("[btn]", "<button type='button' class='btn btn-danger danger-border' data-bs-toggle='tooltip' data-bs-placement='top' title='「再」啦！'>").replace("[/btn]", "</button>")
                    sentenceLIST.append(checkSTR)
                    app.logger.info("變成{}".format("".join(sentenceLIST)))
            else:
                sentenceLIST.append(re.sub(pat, "", i))
        #</Loki 的計算區塊>
        #<ChatGPT 的計算區塊>
        if inputDICT["runLLM"] == True:
            if openai.api_key == "":
                chatGPTResultSTR = "沒有設定可用的 token..."
                tokenCount = 0
            else:
                ChatGPTResponse = openai.ChatCompletion.create(model    ="gpt-3.5-turbo",
                                                               max_tokens=128,
                                                               temperature=0.5,
                                                               messages =[{"role": "system", "content": "你是個中文文法專家"},
                                                                          {"role": "assistant", "content": "請讀這篇文章：「{}」".format(inputSTR)},
                                                                          {"role": "user", "content": "檢查這篇文章裡是否有錯別字。"}
                                                                         ],
                                                               )
                chatGPTResultSTR = ChatGPTResponse.choices[0].message.content
                tokenCount = ChatGPTResponse.usage.total_tokens
            #</ChatGPT 的計算區塊>
            response = jsonify({"checkResult":"".join(sentenceLIST), "chatgptResult":"ChatGPT 回覆>>用了 {} token 計算後得出…<br>{}<br><br>估計費用為{}元".format(tokenCount, chatGPTResultSTR, float(tokenCount)*0.002/1000)})    #將最終結果以 jsonify() 包裝後回傳到前端 .js
        else:
            response = jsonify({"checkResult":"".join(sentenceLIST)})    #將最終結果以 jsonify() 包裝後回傳到前端 .js

        return response

if __name__ == "__main__":
    app.run(debug=True)
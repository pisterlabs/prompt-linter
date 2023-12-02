
from uiautomation import WindowControl,MenuControl
import uiautomation as automation
import openai
import getapi
import mate



openai.api_key = "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"

wx = WindowControl( Name='深圳内部小分队')
print(wx)
#切换到窗口
wx.SwitchToThisWindow()

def call_back(result,json_obj,state):
    outstr = result["outstr"]
    #把字符串outstr 中的 \n 换成 {Shift}{Enter}
    outstr = outstr.replace("\n","{Shift}{Enter}")


    wx.EditControl(Name="输入").SendKeys(outstr)
    wx.ButtonControl(Name="发送(S)").Click()



def ask(question, context=None):
    if context is None:
        prompt = question
    else:
        prompt = f"{context}\n{question}"
    try:
        response = openai.Completion.create(
            #engine="davinci",
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
            frequency_penalty=0.5,
            presence_penalty=0.5,
        )
        if context is None:
            return response.choices[0].text.strip()
        else:
            return response.choices[0].text.replace(context, "").strip()
    
    except:
        pass
    
    
context = None



def chat(question,context=None):
    answer = ask(question, context=context)
    context = answer
    return answer









#hw = wx.ListControl(Name='会话')

context = None
lodmess = []
bot_name = "机器人-小哆"

def get_message(chobj): #信息的单词是 
    ilist = chobj.GetFirstChildControl().GetChildren()
    if len(ilist) ==0:
        return [0]
    else:
     return [chobj.GetFirstChildControl().GetChildren()[0].Name,chobj.Name]


context = "你好"
get_api = getapi.get_api(call_back)
while True:

    try:
            last_msg = wx.ListControl(Name = "消息").GetChildren()


            sendName = get_message(last_msg[-1])[0]
            sendMsg = get_message(last_msg[-1])[1]
            #查找 sendMsg 是否在开头包含 @+bot_name 字符串
            if sendMsg.startswith("@"+bot_name):
                print("bot_name")
                #过滤掉@+bot_name
                sendMsg = sendMsg.replace("@"+bot_name,"")
                #查看 lodmess 是否有 sendMsg
                if sendMsg in lodmess:
                    print("已经回复过了")
                    continue  #如果有，则跳过
                else:
                    lodmess.append(sendMsg)
                    #看 lodmess 的长度是否大于 10 ,如果大于，则删除第一个
                    if len(lodmess) > 10:
                        lodmess.pop(0)
                        



                #如果是，则调用gpt
                #获取回复
                reply = "亲爱的"+sendName+"。。。"
                #回复消息
                wx.EditControl(Name="输入").SendKeys(reply)
                wx.ButtonControl(Name="发送(S)").Click()
                print("回复成功")
                #sendMsg = '\u2005你们好'
                #过了掉\u2005
                sendMsg = sendMsg.replace('\u2005','') 

                
                tcommand = mate.get_mate(sendMsg)

                if tcommand != False:
                    get_api.add_get_api(tcommand[0])
                else:
                    #getchat = chat(sendMsg,context)
                    #wx.EditControl(Name="输入").SendKeys(getchat)
                    wx.EditControl(Name="输入").SendKeys("今天墙太高，翻的超时了！")
                    wx.ButtonControl(Name="发送(S)").Click()





            pass
    except:
        pass
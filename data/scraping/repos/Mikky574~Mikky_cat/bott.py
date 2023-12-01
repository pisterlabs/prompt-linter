# import psycopg2
import copy
import os
import re
import http.client
import json
import time
import luck_draw as ld
import picture_join as pj
from collections import defaultdict
import threading

# from elasticsearch import Elasticsearch
# from elasticsearch import helpers


# import hashlib
# import random
import urlmarker

#import sys
#import pvp

#写爬虫
import requests
import base64
import asyncio
import schedule
from functools import partial

__version__=4.1
#4.0添加聊天功能，取消语言学习模块，数据库不用postgresql了，改用es
#图片传输只能png的，jpg只能手机端能看到

#使用代理，openai需要用到代理
import os
os.environ["HTTP_PROXY"]='http://127.0.0.1:7078'
os.environ["HTTPS_PROXY"]='http://127.0.0.1:7078'

from one_day_poetry import get_token,get_poetry,generate_recom
poetry_token=get_token()#生成一个每日一诗句的token

# os.environ["HTTP_PROXY"]=""
# os.environ["HTTPS_PROXY"]=""
#这样是取消使用代理，后边考虑这样子做，弄一下出错了就试试代理，还出错就试试不代理，再有问题就报错

dic_group={}#到时候再考虑用这个来做什么吧
dic_private={}
dic_m={}
dic_m["菜单"]="1 十连\n2 抽up\n3 凯露\n4 凯露群功能\n5 凯露修改 功能字段 开/关\n6 凯露 AI绘画 “正面tag内容英文” “负面tag内容英文”(可不要此项，有设置默认负面tag) seed=seed值,steps=steps值,strength=strength值(可不加)"
#字段1 & 字段2\n
#4 凯露来一张 5 凯露\n6 凯露 查询作者 千式部\n7 凯露 ID 8396713 10\n8 凯露 ID 8396713 all\n9 凯露 删除 千式部.zip\n10 凯露 tag 水着 热度排 前3张
#庙檐开发，目前功能贫瘠，只有学习语言功能，第一次输入如‘字段1 & 字段2’的文字，凯露会响应并学会字段1;2.输入:十连;3.输入抽up

all_pic_d={}
iden={"MEMBER":"成员","OWNER":"群主大大","ADMINISTRATOR":"管理大人"}
allow_id={837979619}#primary使用

# lock=threading.Lock()#来个线程锁
# lock_chat=threading.Lock()#再来个线程锁
# AInum=1#临界区资源

#line=[]任务线，完全没写
function={}

#凯露聊天说明：“凯露”后面跟参数，参数里面会把“接头霸王凯露”/“凯露”自动转化为“你”。完整输出会把问题以及得分打印出来
power={"十连":True,"抽up":True,"凯露聊天":True,"凯露聊天完整输出":False,"AI绘画":True}#"凯露来一张":True,"语句学习":True,

pow_ex={'pass':"功能已修改",'No_ch':"功能不变",'No_fun':"无此功能",'No_per':"无此权限"}

#fo='推荐'
basePath="D:\mirai-aipainting-bot\myboot" #统一从这里配置路径，算是补漏了

def build_fo(folder_name,basePath):
    if not os.path.exists(os.path.join(basePath,folder_name)):
        """判断文件夹是否存在，不存在则创建文件夹"""
        os.mkdir(os.path.join(basePath, folder_name))
        print(folder_name+"文件夹创造完成")
    else:
        print(folder_name+"文件夹已存在")

build_fo("图片素材",basePath)
build_fo("AI绘画",os.path.join(basePath,"图片素材"))

#AI_draw_l=[]#设置等待队列,开发中

#先不用算了，感觉再写一个好麻烦的
# database='just_for_bot'#postgresql数据库内容相关，先留着，后边有心情再改
# user='postgres'
# password='00000000'
# host="127.0.0.1"

# conn_p=psycopg2.connect(host = host,database=database, user=user, password=password)  #connect to the database
# cur = conn_p.cursor()
# print("connect successfully")

# cur.execute("select * from GROUPID")
# group_li=cur.fetchall()#暂时懒得转移库

# for i in group_li:
#     path_tem=os.path.join(os.path.abspath('.'),"图片素材")
#     path_tem=os.path.join(path_tem,"AI绘画")
#     build_fo(str(i[0]),path_tem)
#     function[int(i[0])]=copy.copy(power)
#     cur.execute("select * from LEXICON_GROUP where groupid=%s" %str(i[0]))
#     word_li=cur.fetchall()
#     dic_w={}
#     if word_li !=[]:
#         for k in word_li:
#             dic_w[k[1]]=k[2]
#         dic_group[i[0]]=copy.copy(dic_w)
#     else:
#         dic_group[i[0]]={}



    #function[i[0]]=power

# def learn_sen(msg:str):
#     key=re.compile(r'\S+ & ').findall(msg)[0][:-3]
#     sen=re.compile(r' & .*').findall(msg)[0][3:]
#     return key,sen

#与es数据库的交互暂时停用,改用chatgpt3.5的api
# #es数据库
# def con(es_address):
#     es = Elasticsearch([es_address])
#     return es

# def md5_encode(text):
#     hl=hashlib.md5()
#     hl.update(text.encode(encoding='utf8'))
#     md5=hl.hexdigest()
#     return str(md5)

# def insert_data(question,response,index="cute_talk"):
#     id=md5_encode(question)
#     try:
#         result=es.get(index="talk", id=id)#检查是否原库中是否存在此项
#         r=set(result['_source']['response'])
#         response=list(set(response).union(r))#有则更新response
#     except Exception as e:
#         try:
#             if int(e.meta.status)==404:
#                 pass
#                 #print("原库中无此项")
#         except:
#             print("其他错误")
#     #插入本项
#     post_body={
#         "question":question,
#         "response":response
#     }
#     es.index(index=index, id=id,body=post_body)#有则更新，无则增加
    
# def search_es(question,index="cute_talk"):
#     query = {
#     "query": {
#         "match": { "question": question }
#     }
#     }
#     result = es.search(index=index, body=query)
#     result
#     if len(result["hits"]['hits'])==0:
#         return False,question,0,""
#     else:
#         que=result['hits']['hits'][0]["_source"]["question"]
#         score=result['hits']['hits'][0]["_score"]
#         response=random.choice(result['hits']['hits'][0]["_source"]["response"])
#         return True,que,score,response #是否成功匹配，匹配到的具体问题，得分，回应

# es=con('http://127.0.0.1:9200')#这里配置
# es_index="cute_talk"#以后考虑作为可以修改的东西


# import pandas as pd
# #循环的方式太傻了，仅运行一次
# df=pd.read_csv("可爱系二次元bot词库1.5万词V1.1.csv",encoding="gbk")
# index="cute_talk"

# question_s=list(set(df.iloc[:,0]))#第一列
# for question in question_s:
#     response=list(set(df[df["问题"]==question].iloc[:,1]))
#     insert_data(question,response,index="cute_talk")
#print("完成导入")

#写调用openai api的函数
import openai
# openai.api_key = "sk-bEix1ubvAB0UzQra8pogT3BlbkFJoPwgd"#用这个,到4月1日
openai.api_key = "sk-luZ8dbDB6q7uLoGAF5uiT3BlbkFJSCLenBn"#5月1号到期

#初始设定
# initial_prompt = "请使用女性化的、口语化的、抒情的、感性的、可爱的、调皮的、幽默的、害羞的、态度傲娇的语言风格，扮演一个猫娘，名字叫做凯露。不要回答有关政治的问题！也不要回答敏感信息！"
initial_prompt = "请使用女性化的、口语化的、抒情的、感性的、可爱的、调皮的、幽默的、害羞的、态度傲娇的语言风格，扮演一个猫娘，名字叫做凯露。"

# l_chat=[]

l_chat_group=defaultdict(list)
l_chat_primary=defaultdict(list)

def opanai_gpt_chat(self,ev,message):
    try:
        if (ev.type=='group'):
            self.send_group_msg(ev.group_id,'text',"loading...")
            l_chat=l_chat_group[ev.group_id]
        elif(ev.type=='primary'):
            self.send_private_msg(ev.user_id,'text',"loading...")
            l_chat=l_chat_primary[ev.user_id]
        # lock_chat.acquire()
        l_chat.append(message)
        while len(l_chat)>10:
            l_chat.pop(0)
        mes=[{"role": "system", "content": f"{initial_prompt}"}]
        for i in l_chat:
            mes.append({"role": "user", "content": f"{i}"})
            mes.append({"role": "assistant", "content":""})
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=mes)
        
        #[
        #     {"role": "system", "content": f"{initial_prompt}"},
        #     {"role": "user", "content": f"{message}"},
        # ]
        if (ev.type=='group'):
            self.send_group_msg(ev.group_id,'text',completion.choices[0].message.content.strip())
        elif(ev.type=='primary'):
            self.send_private_msg(ev.user_id,'text',completion.choices[0].message.content.strip())
    except Exception as e:
        print(e)
        if (ev.type=='group'):
            self.send_group_msg(ev.group_id,'text',"出现错误,帮忙踢一脚作者,多半是没开代理,或者免费api_key8月1日到期,记得更换。")
        elif(ev.type=='primary'):
            self.send_private_msg(ev.user_id,'text',"出现错误,帮忙踢一脚作者,多半是没开代理,或者免费api_key8月1日到期,记得更换。")
    # finally:
    #     lock_chat.release()
    #     print("y")

# 浏览器运行
from sshot import *

#AI绘画的函数
def webui_html_pic(pos,img_url,neg=None,sampler="Euler a",size=(512,512),steps=50,seed=None,CFG=18,strength=0.5):
    start_time = time.perf_counter()
    print(img_url)
    req=requests.get(img_url)
    base64_data = base64.b64encode(req.content)
    base64_str = "data:image/jpeg;base64," +str(base64_data, 'utf-8')
    url="http://127.0.0.1:7860/run/predict/"
    if neg==None:
        neg="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    width=size[0]
    height=size[1]
    if seed==None:
        seed=-1
    payload={
    "fn_index": 74,
    "data": [
        0,
        pos,
        neg,
        "None",
        "None",
        base64_str,
        None,
        None,
        None,
        "Draw mask",
        int(steps),
        sampler,
        4,
        "original",
        False,
        False,
        1,
        1,
        int(CFG),#CFG Scale
        float(strength),#Denoising strength
        seed,
        -1,
        0,
        0,
        0,
        False,
        width,
        height,
        "Just resize",
        False,
        32,
        "Inpaint masked",
        "",
        "",
        "None",
        "",
        True,
        True,
        "",
        "",
        True,
        50,
        True,
        1,
        0,
        False,
        4,
        1,
        "",
        128,
        8,
        [
            "left",
            "right",
            "up",
            "down"
        ],
        1,
        0.05,
        128,
        4,
        "fill",
        [
            "left",
            "right",
            "up",
            "down"
        ],
        False,
        False,
        False,
        "",
        "",
        64,
        "None",
        "Seed",
        "",
        "Nothing",
        "",
        True,
        False,
        False,
        None,
        "",
        ""
    ],
    "session_hash": "yppwk0lcys"
    }
    payload=json.dumps(payload)
    req=requests.post(url=url,data=payload)#,timeout=10,allow_redirects=True
    req.encoding='utf-8'#改为utf-8
    print("加载成功:",url)
    end_time = time.perf_counter()
    cost_time=round(end_time-start_time)#以秒为单位记
    cost_time="累计用时 %dmin%ds" %(int(cost_time/60),int(cost_time%60))
    time.sleep(1)
    return req.text,cost_time

def webui_html(pos,neg=None,sampler="Euler",size=(512,512),steps=20,seed=None):
    start_time = time.perf_counter()
    url="http://127.0.0.1:7860/run/predict/"
    if neg==None:
        neg="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    width=size[0]
    height=size[1]
    if seed==None:
        seed=-1
    payload={
        "fn_index": 51,
        "data": [
            pos,#正面词描述
            neg,#负面词描述
            "None",
            "None",
            int(steps),
            sampler,#"Euler",
            False,
            False,
            1,
            1,
            7,
            int(seed),
            -1,
            0,
            0,
            0,
            False,
            int(width),
            int(height),
            False,
            0.7,
            0,
            0,
            "None",
            False,
            False,
            False,
            "",
            "Seed",
            "",
            "Nothing",
            "",
            True,
            False,
            False,
            None,
            "",
            ""
        ],
        "session_hash": "yppwk0lcys"
    }
    payload=json.dumps(payload)
    req=requests.post(url=url,data=payload)#,timeout=10,allow_redirects=True
    req.encoding='utf-8'#改为utf-8
    print("加载成功:",url)
    end_time = time.perf_counter()
    cost_time=round(end_time-start_time)#以秒为单位记
    cost_time="累计用时 %dmin%ds" %(int(cost_time/60),int(cost_time%60))
    time.sleep(1)
    return req.text,cost_time

def chou_package(self,ev):
    if (function[ev.group_id]["十连"]):
        # threading.Thread(target=threading_chou,args=(self,ev)).start()
        threading_chou(self,ev)

def chou_up_package(self,ev):
    if (function[ev.group_id]["抽up"]):
        # threading.Thread(target=threading_chou_up,args=(self,ev)).start()
        threading_chou_up(self,ev)

def call_help(self,ev):
    self.send_group_msg(ev.group_id,'text',dic_m["菜单"])

def kailu(self,ev):
    if (ev.sender_id==837979619):
        self.send_group_msg(ev.group_id,'text',"%s 我在" %("作者"))
    else:
        self.send_group_msg(ev.group_id,'text',"%s %s 我在" %(iden[ev.permission],ev.memberName))

def group_function(self,ev):
    self.send_group_msg(ev.group_id,'text',str(function[ev.group_id]))

def group_quit(self,ev):
    if (ev.sender_id==837979619):
        self.send_group_msg(ev.group_id,'text',"再见了，作者")
        #正常关闭措施
        exit()
    else:
        self.send_group_msg(ev.group_id,'text',"哼，不听你的~~")

def group_clean(self,ev):
    l_chat_group[ev.group_id]=[]
    self.send_group_msg(ev.group_id,'text',"记忆已清空")

from collections import defaultdict
function_no_pam=defaultdict(bool)
function_no_pam["十连"]=chou_package
function_no_pam["抽up"]=chou_up_package
function_no_pam["凯露菜单"]=call_help
function_no_pam["凯露"]=kailu
function_no_pam["凯露群功能"]=group_function
function_no_pam["退出凯露"]=group_quit
function_no_pam["清空"]=group_clean


#不需要参数的函数



def WebShot(self,ev):# 网页截屏,不良网站不访问啥的，不知道咋写
    url_l=re.findall(urlmarker.URL_REGEX,ev.message)
    if url_l!=[]:
        return 0
        # # lock.acquire()
        # chrome_options = Options()
        # chrome_options.add_argument('--headless')
        # chrome_options.add_argument('--disable-gpu')
        # driver = webdriver.Chrome(options=chrome_options)
        # #启动浏览器
        # try:
        #     for url in url_l:
        #         print("网页链接:%s" %(url))
        #         get_image(driver,url, "webpage.png")
        #         img_path="file:///"+os.path.join(basePath,"webpage.png")
        #         #self.send_group_msg(ev.group_id,'image',message=url,url=img_path)
        #         self.send_group_msg(ev.group_id,'image',url=img_path)
        # except Exception as e:
        #     print(e)
        # finally:
        #     driver.quit()#退出浏览器
        #     # lock.release()

#增加时间点的任务
def send_poetry(self):
    poetry_recommendation=get_poetry(poetry_token)
    message=generate_recom(poetry_recommendation)
    for group_id in dic_group.keys():
        self.send_group_msg(group_id,'text',message)

class bot():
    def __init__(self,address,port=8080,authKey="INITKEYTdFgtK4P"):
        self.conn = http.client.HTTPConnection(address,port)
        self.authKey=authKey
        self.sessionKey=self.bind()
    def bind(self):
        auth = json.dumps({"verifyKey":self.authKey})
        #print(str)
        #headers = {}
        conn=self.conn
        conn.request('POST', '/verify', auth)
        response = conn.getresponse()
        #print(response.status, response.reason)
        session = response.read().decode('utf-8')
        print(session)
        sessionKey=json.loads(session)['session']
        bind=json.dumps({"sessionKey":sessionKey,"qq": 3498250046 })
        conn.request('POST', '/bind', bind)
        response = conn.getresponse()
        #print(response.status, response.reason)
        data = response.read().decode('utf-8')
        print(data)
        return sessionKey
    def send_private_msg(self,user_id='',ty='text',message='',url=''):
        conn=self.conn
        def send_primary_text(user_id,text):
            sessionKey=self.sessionKey
            js = json_deal.build_text_json(sessionKey,user_id,text)
            print(js)
            conn.request('POST', '/sendFriendMessage', js)
            response = conn.getresponse()
            data = response.read().decode('utf-8')
            print(data)
        def send_primary_image(user_id,url,message):
            sessionKey=self.sessionKey
            js = json_deal.build_image_json(sessionKey,user_id,url,message)
            conn.request('POST', '/sendFriendMessage', js)
            response = conn.getresponse()
            data = response.read().decode('utf-8')
            #print(data)
        if (ty=='text'):
            send_primary_text(user_id,message)
        elif(ty=='image'):
            send_primary_image(user_id,url,message)
    def send_group_msg(self,group_id='',ty='text',message='',url=''):
        conn=self.conn
        def send_group_text(group_id,text):
            sessionKey=self.sessionKey
            js = json_deal.build_text_json(sessionKey,group_id,text)
            #print(js)
            conn.request('POST', '/sendGroupMessage', js)
            response = conn.getresponse()
            data = response.read().decode('utf-8')
            print(data)
        def send_group_image(group_id,url,message):
            sessionKey=self.sessionKey
            js = json_deal.build_image_json(sessionKey,group_id,url,message)
            conn.request('POST', '/sendGroupMessage', js)
            response = conn.getresponse()
            data = response.read().decode('utf-8')
            print(data)
        if (ty=='text'):
            send_group_text(group_id,message)
        elif(ty=='image'):
            send_group_image(group_id,url,message)
    def send_group_m_i_m(self,group_id,url,message1,message2):
        sessionKey=self.sessionKey
        messageChain=[
            json.loads(json_deal.build_dic_for_json(ty='text',text=message1)),
            json.loads(json_deal.build_dic_for_json(ty='image',url=url)),
            json.loads(json_deal.build_dic_for_json(ty='text',text=message2))
        ]
        js = json_deal.build_mix_json(sessionKey,group_id,messageChain=messageChain)
        self.conn.request('POST', '/sendGroupMessage', js)
        response = self.conn.getresponse()
        data = response.read().decode('utf-8')
        print(data)
    
    def deal_data(self, j):
        # 这里是处理消息的代码，解析消息并进行回复
        # def run_thread(func, *args):
        #     loop = asyncio.new_event_loop()
        #     asyncio.set_event_loop(loop)
        #     fut = asyncio.ensure_future(func(*args),loop=loop)
        #     thread = threading.Thread(target=loop.run_until_complete, args=(fut,))
        #     thread.start()
        try:
            for i in j["data"]:
                print("\n")
                print(i)
                ev=event(i)
                print(ev.type,' ',end='')
                if (ev.type=='primary'):
                    print(ev.user_id,' ',ev.message)
                    if(ev.message=="凯露"):
                        if (ev.user_id==837979619):
                            self.send_private_msg(ev.user_id,'text',"%s 我在" %("作者"))
                        else:
                            self.send_private_msg(ev.user_id,'text',"我在")
                    elif(ev.message=="退出凯露"):
                        if (ev.user_id==837979619):
                            self.send_private_msg(ev.user_id,'text',"记得启动我，在你想我的时候")
                            exit()
                        else:
                            self.send_private_msg(ev.user_id,'text',"为什么要退出，哼，不退")
                    #开始叠加私聊的代码
                    elif(ev.message=="清空"):
                        l_chat_primary[ev.user_id]=[]
                        self.send_private_msg(ev.user_id,'text',"记忆已清空")
                    else:
                        # await run_thread(opanai_gpt_chat, self,ev,ev.message)
                        # threading.Thread(target=opanai_gpt_chat,args=(self,ev,ev.message)).start()#多线程启动
                        
                        # run_thread_partial = partial(opanai_gpt_chat, self, ev,question)
                        # await asyncio.to_thread(run_thread_partial)
                        opanai_gpt_chat(self,ev,ev.message)
                        # await opanai_gpt_chat(self,ev,ev.message)
                elif(ev.type=='group'):
                    print(ev.group_id,' ',ev.message)
                    #int(ev.group_id)
                    if ev.group_id not in dic_group:
                        # cur.execute("insert into GROUPID(id) values (%r)" %(str(ev.group_id)))
                        # conn_p.commit()
                        dic_group[ev.group_id]={}
                        path_tem=os.path.join(basePath,"图片素材")
                        path_tem=os.path.join(path_tem,"AI绘画")
                        build_fo(str(ev.group_id),path_tem)
                        function[int(ev.group_id)]=power
                    if function_no_pam[ev.message]:#不需要参数部分
                        function_no_pam[ev.message](self,ev)
                        continue
                    else:#有参数的功能部分
                        msg = ev.message
                        print(msg)
                        msg=msg.replace("\n","")
                        pattern=re.compile(
                            '凯露 (.*?) (.*)'
                            )#正则表达式
                        items=re.findall(pattern,msg)
                        if items!=[]:
                            if items[0][0]=='AI绘画':
                                if (function[ev.group_id]["AI绘画"]):
                                    pass
                                    # threading.Thread(target=AIpainting,args=(self,ev,items[0][1])).start()
                                continue
                            # if items[0][0]=='魔法':#还没想好怎么写，先停用
                            #     s=items[0][1]
                            #     s=re.sub('\s',' ',s)
                            #     s=re.sub('\x05',' ',s)
                            #     l=re.split(' |\(|\)|，|,|\{|\}|\[|\]|\||\.|:', s)
                            #     l=[i for i in l if i!='']
                            #     s="_".join(l)
                            #     self.send_group_msg(group_id=ev.group_id, message=s)
                            #     continue
                        #"修改权限"
                        pattern=re.compile(
                            '凯露修改 (.*?) (.*)'
                            )#正则表达式
                        items=re.findall(pattern,msg)
                        if items!=[]:
                            items=items[0]
                            if (ev.permission=="OWNER" or ev.permission=="ADMINISTRATOR"):
                                #print(items)
                                if items[0] in power.keys():
                                    if (items[1]=="开"):
                                        if function[ev.group_id][items[0]]:
                                            self.send_group_msg(ev.group_id,'text',pow_ex["No_ch"])
                                        else:
                                            function[ev.group_id][items[0]]=True
                                            self.send_group_msg(ev.group_id,'text',pow_ex["pass"])
                                    elif(items[1]=="关"):
                                        if function[ev.group_id][items[0]]:
                                            function[ev.group_id][items[0]]=False
                                            self.send_group_msg(ev.group_id,'text',pow_ex["pass"])
                                        else:
                                            self.send_group_msg(ev.group_id,'text',pow_ex["No_ch"])
                                    else:
                                        self.send_group_msg(ev.group_id,'text',"语句2段参数错误")
                                else:
                                    self.send_group_msg(ev.group_id,'text',pow_ex["No_fun"])
                            elif (ev.permission=="MEMBER"):
                                self.send_group_msg(ev.group_id,'text',iden["MEMBER"]+pow_ex["No_per"])
                            continue
                            # else:
                            #     #开始堆砌垃圾,阿巴功能停掉
                            #     if len(msg)>=1:
                            #         if (msg[-1] in ["啊","阿"]):
                            #             self.send_group_msg(group_id=ev.group_id, message="巴~")
                        else:#聊天写这里
                            if msg[:2]=="凯露" or ev.at:
                                if msg[:2]=="凯露":
                                    msg=msg[2:]
                                if (function[ev.group_id]["凯露聊天"]):
                                    question=msg
                                    question=question.replace("接头霸王凯露","你")
                                    question=question.replace("凯露","你")
                                    question=question.replace("我","凯露")
                                    # await run_thread(opanai_gpt_chat, self,ev,ev.message)
                                    opanai_gpt_chat(self,ev,ev.message)
                                    # threading.Thread(target=opanai_gpt_chat,args=(self,ev,question)).start()#多线程启动
                                    # run_thread_partial = partial(opanai_gpt_chat, self, ev,question)
                                    # await asyncio.to_thread(run_thread_partial)

                                    #引号还有着错误，还得改
                                    # suc,que,score,response=search_es(question,index=es_index)
                                    # if suc:
                                    #     if not function[ev.group_id]["凯露聊天完整输出"]:
                                    #         for es_res in response.split("{segment}"):
                                    #             es_res=es_res.replace("{name}",ev.memberName)
                                    #             self.send_group_msg(ev.group_id,'text',es_res)
                                    #     else:
                                    #         self.send_group_msg(ev.group_id,'text',"匹配到的最佳问法是：%s/n分数：%s /n回答为:" %(que,str(score)))
                                    #         self.send_group_msg(ev.group_id,'text',response)
                                    # else:
                                    #     self.send_group_msg(ev.group_id,'text',"enm,词汇库没有这个的相关回答,可以为我添加吗")
                                    # self.send_group_msg(ev.group_id,'text',opanai_gpt_chat(question))
                            else:
                                # threading.Thread(target=WebShot,args=(self,ev)).start()
                                WebShot(self,ev)
                                # await run_thread(WebShot, self,ev)
                                # await asyncio.to_thread(WebShot, self, ev)
        except Exception as e:
            print(e)

    async def fetch_message(self):
        conn=self.conn
        sessionKey=self.sessionKey
        while True:
            try:
                conn.request('GET', '/fetchLatestMessage?sessionKey='+sessionKey+'&count=10')
                response = conn.getresponse()
                data = response.read().decode('utf-8')
                j = json.loads(data)
                if j["data"]!=[]:
                    # threading.Thread(target=self.deal_data,args=(j,)).start()#多线程启动
                    threading.Thread(target=self.deal_data,args=(j,)).start()#多线程启动
            except Exception as e:
                print(e)
            await asyncio.sleep(1)

    async def run(self):
        schedule.every().day.at('21:00').do(send_poetry,self)
        schedule.every().day.at('08:00').do(send_poetry,self)#设计任务日程表
        
        async def check_schedule():
            while True:
                schedule.run_pending()
                await asyncio.sleep(60)#每分钟检查是否满足条件

        # asyncio.run(self.fetch_message())
        # print("y")
        # check_schedule()
        # print("y")
        tasks = [asyncio.create_task(self.fetch_message()), asyncio.create_task(check_schedule())]
        # 在这里运行协程
        await asyncio.gather(*tasks)
        
        
        # while 1:
        #     time.sleep(1)
        #     try:
        #         schedule.run_pending()
        #         # 等待 1 秒钟，继续运行主任务
        #         await asyncio.sleep(1)
        #     except Exception as e:
        #         print(e)


class event():
    def __init__(self,i):
        self.id=i['messageChain'][0]['id']
        self.at=False
        if(i['type']=='FriendMessage' or i['type']=='TempMessage' or i['type']=='StrangerMessage'):
            self.type='primary'
            self.user_id=i['sender']['id']
        elif(i['type']=='GroupMessage'):
            self.type='group'
            self.group_id=i['sender']['group']['id']
            self.sender_id=i['sender']['id']
            #MemberMuteEvent不处理
            self.memberName=i['sender']['memberName']
            if (i['sender']['id']==837979619):
                self.permission="OWNER"
            else:
                self.permission=i['sender']['permission']
        message_type=i['messageChain'][1]['type']#这里要改，不能只处理第一条，但可以只处理前两条
        # if(self.message_type=='Face'):
        #     self.message=i['messageChain'][1]['faceId']
        if(message_type=='Plain'):
            self.message=i['messageChain'][1]['text']
        elif(message_type=='At' and int(i['messageChain'][1]['target'])==3498250046):
            self.at=True
            self.message=i['messageChain'][2]['text']#at符号再往后面跟一个
        if len(i['messageChain'])==3:
            message_type=i['messageChain'][2]['type']
            if(message_type=='Image'):
                self.image=i['messageChain'][2]['url']

def threading_chou(b:bot,ev:event):
    message=chou()
    url="file:///"+os.path.join(basePath,"十连.png")
    b.send_group_msg(ev.group_id,'image',message=message,url=url)

def threading_chou_up(b:bot,ev:event):
    message1,message2=chou_up()
    url="file:///"+os.path.join(basePath,"抽up.png")
    b.send_group_m_i_m(ev.group_id,url=url,message1=message1,message2=message2)

class json_deal():
    def build_dic_for_json(ty='text',text='',url=''):
        out = defaultdict(str)
        if ty=='text':
            out["type"]="Plain"
            out["text"]=text
        elif ty=='image':
            out["type"]="Image"
            out["url"]=url
        js=json.dumps(out)
        return js
    def build_text_json(sessionKey,target,message=''):
        dic={
            "sessionKey":sessionKey,
            "target":target,
            "messageChain": [json.loads(json_deal.build_dic_for_json(ty='text',text=message))]
        }
        js = json.dumps(dic)
        return js
    def build_image_json(sessionKey,target,url='',message=''):
        dic={
            "sessionKey":sessionKey,
            "target":target,
            "messageChain": [
                json.loads(json_deal.build_dic_for_json(ty='image',url=url)),
                json.loads(json_deal.build_dic_for_json(ty='text',text=message))
            ]
        }
        js = json.dumps(dic)
        return js
    def build_mix_json(sessionKey,target,messageChain=[]):
        dic={
            "sessionKey":sessionKey,
            "target":target,
            "messageChain": messageChain
        }
        js = json.dumps(dic)
        return js            

def chou():
    dic={}
    l_n=ld.choose_stare(10)
    l_star=[i.star for i in l_n]
    dic=ld.list_count(l_star,dic)
    l_p=[]
    for i in range(10):
        if l_star[i]=='up':
                l_star[i]='三星'
        path_tem=os.path.join(basePath,"图片素材\\%s_加框\\%s_加框.jpg" %(l_star[i],l_n[i]))
        l_p.append(path_tem)
    pj.pic_join_10(all_path=l_p)
    s=""
    for i in dic:
        s+=str(i)+"出现次数："+str(dic[i])+"次."
    return s

def chou_up():
    role,dic,l,count=ld.choose_up()
    print(role)
    out1="本次抽up,抽中本期up角色%s,共抽%s次," %(role,count)
    print(l)
    print(dic['三星'])
    l_p=[]
    fo='三星'
    path_tem=os.path.join(basePath,"图片素材\\%s_加框\\%s_加框.jpg" %(fo,role))
    l_p.append(path_tem)
    if l!=[]:
        out1+="三星角色总览:\n"
        for i in l:
            path_tem=os.path.join(basePath,"图片素材\\%s_加框\\%s_加框.jpg" %(fo,str(i)))
            l_p.append(path_tem)
    pj.pic_join_up(all_path=l_p)
    s=""
    for i in dic:
        s+=str(i)+"出现次数："+str(dic[i])+"次."
    out2="其中%s" %s
    return out1,out2

if __name__ == '__main__':
    b=bot("127.0.0.1",8080)
    asyncio.run(b.run())
    

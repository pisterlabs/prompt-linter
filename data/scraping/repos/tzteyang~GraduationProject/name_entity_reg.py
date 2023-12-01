# -*- coding: utf8 -*-
import json
import requests
import numpy as np
import time
from hanlp_restful import HanLPClient
# from info_extract import openai_query

# def name_reg_quick(text: str):
#     nlp = HanLP.newSegment().enableNameRecognize(True)
#     doc = nlp.seg(text)
#     name_entitys = [elem.word for elem in doc if elem.nature.toString() == 'nr']
#     print(name_entitys)
#     return name_entitys


def name_reg_texsamrt(text: str):
    obj = {
        "str": text,
        "options":
        {
            "input_spec":{"lang":"auto"},
            "word_seg":{"enable":True},
            "pos_tagging":{"enable":True,"alg":"log_linear"},
            "ner":{"enable":True,"alg":"coarse.lua"},
            "syntactic_parsing":{"enable":False},
            "srl":{"enable":False},
            "text_cat":{"enable":False},
        },
    }
    req_str = json.dumps(obj).encode()
    entitys = []
    try:
        url = "https://texsmart.qq.com/api"
        r = requests.post(url, data=req_str).json()
        # print(r["entity_list"])
        entitys = [elem["str"] for elem in r["entity_list"] if elem["type"]["name"] == "person.generic"]
    except Exception as e:
        print('姓名实体识别Texsmart接口请求异常',str(e))
    # print(entitys)
    return entitys


def name_reg_hanlp(text: str):
    # auth不填则匿名，zh中文，mul多语种
    time.sleep(0.5)
    HanLP = HanLPClient('https://www.hanlp.com/api', auth="MjUzNkBiYnMuaGFubHAuY29tOjNLODZoUWxCeVBBaHVtMFI=", language='zh')
    ret_list = HanLP.parse(text, tasks='ner/msra')["ner/msra"]
    # ret_np = np.array(ret_list)
    entitys = [[entity for entity in ret if "PERSON" in entity] for ret in ret_list]
    name_list = []
    for sep in entitys:
        for entity in sep:
            name_list.append(entity[0])
    # print(name_list)
    return name_list


if __name__ == "__main__":
    # name_reg_hanlp("�机分析和大偏差理论及其在金融保险中的应用电子邮箱：junyan@yzu.edu.cn黄健飞 副教授黄健飞，理学博士，副教授，校特聘教授，硕士生导师。2012年毕业于中国科学院数学与系统科学研究院，获博士学位；2013年至2016年在美国爱荷华大学从事生物统计方法的博士后研究工作。已主持完成国家自然科学基金2项，在研2项。已以第一作者或 2项，在研2项。已以第一作者或通讯作者身份在Genetics和Applied Numerical Mathematics等国际著名SCI期刊发表论文30多篇。目前担任中国仿真学会仿真算法专业委员会委员、江苏省计算数学学会常务理事、美国《数学评论》评论员、Inter.J.Model.Simul.Sci.Comput.")
    # s = "#荣誉奖项  毕恩兵,博士,毕业于上海交通大学。2017年出任上海黎元新能源科技有限公司技术总监。主要研究方向包括新型低成本太阳能电池的工作机理研究、石墨烯和半导体等新型功能材料的开发、高效率太阳能电池器件的制备与应用。2018年2月26日入选2018年度上海市青年英才扬帆计划。"
    # name_reg_hanlp(s)
    # nlp=HanLP.newSegment().enableNameRecognize(True)
    # doc = nlp.seg(s)
    # print(doc)
    pass
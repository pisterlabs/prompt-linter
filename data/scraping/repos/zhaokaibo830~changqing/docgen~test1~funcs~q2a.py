import os
from zipfile import ZipFile
import numpy as np
from lxml import etree
import xml.etree.ElementTree as ET
import zipfile
from docx import Document
import shutil
from langchain.llms.base import LLM
from typing import List, Optional
import requests
import json
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.document_loaders import Docx2txtLoader
from langchain.embeddings import HuggingFaceEmbeddings


class Vicuna(LLM):
    max_token: int = 2048
    temperature: float = 0.8
    top_p = 0.9
    tokenizer: object = None
    model: object = None
    history_len: int = 1024
    # url_llm = "https://u147750-b6ae-2bf49303.neimeng.seetacloud.com:6443/llm"
    url_llm = "https://u147750-92ae-0299e063.neimeng.seetacloud.com:6443/llm"

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "Vicuna"

    def llm(self, prompt: str):
        try:
            content1 = json.dumps({"text": prompt})
            response = requests.request("POST", self.url_llm, data=content1)
            res = response.content.decode('unicode_escape')
            return json.loads(res, strict=False)['response']
        except:
            return "服务器已关闭，请联系服务器管理员"

    def _call(self, prompt: str, stop: Optional[List[str]] = None):
        response = self.llm(prompt)
        return response

def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')

def zip_dirs(*dirs):
    prefix = os.path.commonprefix(dirs)
    with ZipFile('completefile.zip', 'w') as z:
        for d in dirs:
            z.write(d, arcname=os.path.relpath(d, prefix))
            for root, dirs, files in os.walk(d):
                for fn in files:
                    z.write(
                        fp := os.path.join(root, fn),
                        arcname=os.path.relpath(fp, prefix)
                    )


def docx_to_xml(docx_path,xml_save_path):
    """
    :param docx_path:word文档路径
    :param xml_save_path:生成的xml文件保存路径
    :return:
    """
    doc = Document(docx_path)
    body_xml_str = doc._body._element.xml # 获取body中的xml
    body_xml = etree.fromstring(body_xml_str) # 转换成lxml结点
    # print(etree.tounicode(body_xml)) # 打印查看
    mode = 'w'
    with open(xml_save_path, mode,encoding='utf-8') as f:
        # string = string.encode('utf-8')
        f.write(etree.tounicode(body_xml))

def generate_table_description(table_describe_template_path,xml_save_path):

    f=open(table_describe_template_path,encoding='utf-8')
    isfix_tablehead=f.readline()
    table_index=int(f.readline())
    table_describe_template = f.readline()
    table_x_y=[]
    one_x_y=f.readline()
    while one_x_y:
        x,y=int(one_x_y.split(",")[0]),int(one_x_y.split(",")[1])
        table_x_y.append([x,y])
        one_x_y=f.readline()

    tree = ET.parse(xml_save_path)  # 类ElementTree
    root = tree.getroot()  # 类Element
    root_tag=root.tag
    i=len(root_tag)-1

    while True:
        if root_tag[i]=="}":
            break
        i-=1

    prefix=root_tag[:i+1]
    body=root.find(prefix+"body")
    tbl=list(body.findall(prefix+"tbl"))[table_index-1]
    all_rows=list(tbl.findall(prefix+"tr"))
    table_describe_template_prefix=table_describe_template.split("#")[0]
    table_describe_template_suffix=table_describe_template.split("#")[1]

    result=table_describe_template_prefix

    value=[]
    if int(isfix_tablehead)==0:
        # isfix_tablehead为0表示表头不固定，否则表示表头固定
        for r, c in table_x_y:
            temp_value = ""
            all_p = list(all_rows[r - 1].findall(prefix + "tc"))[c - 1].findall(prefix + "p")
            # print("p长度=", len(list(all_p)))

            if len(list(all_p)) == 0:
                temp_value = "没填写内容"
            else:
                for one_p in list(all_p):
                    all_r = list(one_p.findall(prefix + "r"))
                    for one_r in all_r:
                        temp_value = temp_value + one_r.find(prefix + "t").text
            if not temp_value:
                temp_value = "没填写内容"
            value.append(temp_value)
        result += table_describe_template_suffix.format(*value)
        pass
    else:
        for i in range(len(all_rows)-table_x_y[0][0]+1):
            if len(list(all_rows[i+table_x_y[0][0]-1].findall(prefix + "tc"))) == 1:
                break
            if i==0:
                value=[]
                for r,c in table_x_y:
                    temp_value=""
                    all_p = list(all_rows[r - 1 + i].findall(prefix + "tc"))[c - 1].findall(prefix + "p")
                    # print("p长度=", len(list(all_p)))

                    if len(list(all_p)) == 0:
                        temp_value = "没填写内容"
                    else:
                        for one_p in list(all_p):
                            all_r = list(one_p.findall(prefix + "r"))
                            for one_r in all_r:
                                temp_value = temp_value + one_r.find(prefix + "t").text
                    if not temp_value:
                        temp_value = "没填写内容"

                    value.append(temp_value)
            else:
                for j,(r,c) in enumerate(table_x_y):
                    # print("j=",j)
                    all_vMerge = list(all_rows[r - 1 + i].findall(prefix + "tc")[c - 1].find(prefix + "tcPr").findall(
                        prefix + "vMerge"))

                    if len(all_vMerge)>0 and all_vMerge[0].attrib[prefix + "val"]=="continue":
                        continue
                    else:
                        temp_value=""
                        all_p=list(all_rows[r-1+i].findall(prefix+"tc"))[c-1].findall(prefix+"p")
                        # print("p长度=", len(list(all_p)))

                        if len(list(all_p)) == 0:
                            temp_value = "没填写内容"
                        else:
                            for one_p in list(all_p):
                                all_r=list(one_p.findall(prefix+"r"))
                                for one_r in all_r:
                                    temp_value=temp_value+one_r.find(prefix+"t").text
                        if not temp_value:
                            temp_value = "没填写内容"

                        value[j]=temp_value
            arr = np.array(value)
            if (arr == "没填写内容").all():
                break
            # print(value)
            result += table_describe_template_suffix.format(*value)
        if len(list(all_rows[-1].findall(prefix + "tc")))==1:
            temp_value = ""
            all_p = list(all_rows[- 1].findall(prefix + "tc"))[0].findall(prefix + "p")
            # print("p长度=", len(list(all_p)))

            if len(list(all_p)) > 0:
                for one_p in list(all_p):
                    all_r = list(one_p.findall(prefix + "r"))
                    for one_r in all_r:
                        temp_value = temp_value + one_r.find(prefix + "t").text
                result+=temp_value

    return table_index,result

def generate_table2_description(table_describe_template_path,xml_save_path):

    f=open(table_describe_template_path,encoding='utf-8')
    isfix_tablehead=f.readline()
    table_index=int(f.readline())
    table_describe_template = f.readline()
    table_x_y=[]
    one_x_y=f.readline()
    while one_x_y:
        x,y=int(one_x_y.split(",")[0]),int(one_x_y.split(",")[1])
        table_x_y.append([x,y])
        one_x_y=f.readline()

    tree = ET.parse(xml_save_path)  # 类ElementTree
    root = tree.getroot()  # 类Element
    root_tag=root.tag
    i=len(root_tag)-1

    while True:
        if root_tag[i]=="}":
            break
        i-=1

    prefix=root_tag[:i+1]
    body=root.find(prefix+"body")
    tbl=list(body.findall(prefix+"tbl"))[table_index-1]
    all_rows=list(tbl.findall(prefix+"tr"))
    table_describe_template_prefix=table_describe_template.split("#")[0]
    table_describe_template_suffix=table_describe_template.split("#")[1]

    result=table_describe_template_prefix

    value=[]
    if int(isfix_tablehead)==0:
        # isfix_tablehead为0表示表头不固定，否则表示表头固定
        for r, c in table_x_y:
            temp_value = ""
            all_p = list(all_rows[r - 1].findall(prefix + "tc"))[c - 1].findall(prefix + "p")
            # print("p长度=", len(list(all_p)))

            if len(list(all_p)) == 0:
                temp_value = "没填写内容"
            else:
                for one_p in list(all_p):
                    all_r = list(one_p.findall(prefix + "r"))
                    for one_r in all_r:
                        temp_value = temp_value + one_r.find(prefix + "t").text
            if not temp_value:
                temp_value = "没填写内容"
            value.append(temp_value)
        result += table_describe_template_suffix.format(*value)
        pass
    else:
        # print(len(all_rows))
        for i in range(len(all_rows)-table_x_y[0][0]+1):
            count=0 #记录这一列是否被分割
            if len(list(all_rows[i+table_x_y[0][0]-1].findall(prefix + "tc"))) == 1:
                break
            forth_val_2 = ""
            if i==0:
                value=[]

                for r,c in table_x_y:

                    temp_value=""
                    all_p = list(all_rows[r - 1 + i].findall(prefix + "tc"))
                    # print(count)

                    all_p=all_p[c+count - 1].findall(prefix + "p")
                    # print("p长度=", len(list(all_p)))

                    if len(list(all_p)) == 0:
                        temp_value = "没填写内容"
                    else:
                        for one_p in list(all_p):
                            all_r = list(one_p.findall(prefix + "r"))
                            for one_r in all_r:
                                temp_value = temp_value + one_r.find(prefix + "t").text
                    if not temp_value:
                        temp_value = "没填写内容"

                    value.append(temp_value)

                    if c == 4 and len(list(all_rows[r - 1 + i].findall(prefix + "tc")))>7:

                        count+=1
                        all_p = list(all_rows[r - 1 + i].findall(prefix + "tc"))[c + count - 1].findall(prefix + "p")
                        # print("p长度=", len(list(all_p)))

                        if len(list(all_p)) != 0:
                            for one_p in list(all_p):
                                all_r = list(one_p.findall(prefix + "r"))
                                for one_r in all_r:
                                    forth_val_2 = forth_val_2 + one_r.find(prefix + "t").text

            else:
                for j,(r,c) in enumerate(table_x_y):
                    # print("j=",j)
                    all_vMerge = list(all_rows[r - 1 + i].findall(prefix + "tc")[c+count - 1].find(prefix + "tcPr").findall(
                        prefix + "vMerge"))

                    if len(all_vMerge)>0 and all_vMerge[0].attrib[prefix + "val"]=="continue":
                        if c == 4 and len(list(all_rows[r - 1 + i].findall(prefix + "tc")))>7:
                            forth_val_2=""
                        else:
                            continue
                    else:
                        temp_value=""
                        all_p=list(all_rows[r-1+i].findall(prefix+"tc"))[c+count-1].findall(prefix+"p")
                        # print("p长度=", len(list(all_p)))

                        if len(list(all_p)) == 0:
                            temp_value = "没填写内容"
                        else:
                            for one_p in list(all_p):
                                all_r=list(one_p.findall(prefix+"r"))
                                for one_r in all_r:
                                    temp_value=temp_value+one_r.find(prefix+"t").text
                        if not temp_value:
                            temp_value = "没填写内容"

                        value[j] = temp_value
                    if c == 4 and len(list(all_rows[r - 1 + i].findall(prefix + "tc")))>7:
                        count += 1
                        all_p = list(all_rows[r - 1 + i].findall(prefix + "tc"))[c + count - 1].findall(
                            prefix + "p")
                        # print("p长度=", len(list(all_p)))

                        if len(list(all_p)) != 0:
                            for one_p in list(all_p):
                                all_r = list(one_p.findall(prefix + "r"))
                                for one_r in all_r:
                                    forth_val_2 = forth_val_2 + one_r.find(prefix + "t").text

            # print(value)
            arr = np.array(value)
            if (arr == "没填写内容").all():
                break
            # print(value)
            value_temp=value[:]
            value_temp[3]=value_temp[3]+forth_val_2
            result += table_describe_template_suffix.format(*value_temp)
        if len(list(all_rows[-1].findall(prefix + "tc")))==1:
            temp_value = ""
            all_p = list(all_rows[- 1].findall(prefix + "tc"))[0].findall(prefix + "p")
            # print("p长度=", len(list(all_p)))

            if len(list(all_p)) > 0:
                for one_p in list(all_p):
                    all_r = list(one_p.findall(prefix + "r"))
                    for one_r in all_r:
                        temp_value = temp_value + one_r.find(prefix + "t").text
                result+=temp_value

    return table_index,result



def table_describe_to_doc(table_index,table_describe,complete_file_path):
    f=open(complete_file_path+"/word/document.xml",encoding='utf-8')
    file_str=f.read()
    f.close()

    index_temp=table_index
    start=-1
    while index_temp>0:
        start=file_str.find('<w:tbl>',start+1)
        index_temp-=1

    index_temp=table_index
    end=-1
    while index_temp>0:
        end=file_str.find('</w:tbl>',end+1)
        index_temp-=1
    end=end+7
    insertleft="""<w:p>
<w:pPr>
  <w:spacing w:line="360" w:lineRule="auto"/>
  <w:ind w:firstLine="480" w:firstLineChars="200"/>
  <w:rPr>
    <w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/>
    <w:sz w:val="24"/>
    <w:szCs w:val="24"/>
  </w:rPr>
</w:pPr>
<w:bookmarkStart w:id="40" w:name="_Toc83780989"/>
<w:bookmarkStart w:id="41" w:name="_Toc77646121"/>
<w:bookmarkStart w:id="42" w:name="_Toc415414318"/>
<w:r>
  <w:rPr>
    <w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/>
    <w:sz w:val="24"/>
    <w:szCs w:val="24"/>
  </w:rPr>
  <w:t>"""
    insertright="""</w:t>
</w:r>
</w:p>"""
    inserttext=insertleft+table_describe+insertright
    new_file_str=file_str[:start]+inserttext+file_str[end+1:]
    with open(complete_file_path+"/word/document.xml",encoding='utf-8',mode="w") as f:
        f.write(new_file_str)
    pass

def start():

    # os.rename("completefile.docx", "completefile.zip")
    # unzip_file("completefile.zip", "completefile/")
    # os.rename("completefile.zip", "completefile.docx")
    # shutil.copy("completefile/word/document.xml","document.xml")
    #
    # for m in range(18,1,-1):
    #     if m==2:
    #         table_index, table_describe = generate_table2_description("all_table_descibe_template/{}.txt".format(m),
    #                                                                   "document.xml")
    #     else:
    #         table_index, table_describe = generate_table_description("all_table_descibe_template/{}.txt".format(m),
    #                                                                   "document.xml")
    #     table_describe_to_doc(table_index,table_describe,"completefile")
    #
    #
    # ll = os.listdir("completefile")
    # temp = []
    # for ll_one in ll:
    #     temp.append("completefile/" + ll_one)
    # # print(temp)
    # zip_dirs(*temp)
    # os.rename("completefile.zip","target.docx")
    #
    # loader = Docx2txtLoader("target.docx")
    #
    # data = loader.load()
    #
    # path = 'result.txt'
    # mode = 'w'
    # string = data[0].page_content
    # with open(path, mode, encoding='utf-8') as f:
    #     # string = string.encode('utf-8')
    #     f.write(string)

    # 载入大模型
    llm = Vicuna()

    # openai_api_key = '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'

    loader = TextLoader("result.txt", autodetect_encoding=True)

    documents = loader.load()
    # print(documents)
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=50, separator="\n")
    texts = text_splitter.split_documents(documents)

    faiss_index = FAISS.from_documents(texts, HuggingFaceEmbeddings())

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=faiss_index.as_retriever())

    return qa

def ReportQA(qa,question):
        pre = qa.run(question)
        return pre










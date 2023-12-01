import time
import os
import json
import pandas as pd
import pytesseract
import re
import smtplib
import selenium
from datetime import datetime as dt
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from operator import itemgetter
from typing import Union
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

summary_prompt = """给定一个招标文件，我需要你提取以下关键信息。请提取以下字段：
1. "招标人"：发布招标的单位或者个人的名称。
2. "报价"：招标的报价，单位为万元。
3. "联系信息"：包括地址、电话和电子邮件等的联系方式。
4. "项目描述"：包括项目的目标、背景、范围和期限等详细描述。
5. "资格要求"：对参与投标的公司或个人的资格要求。
6. "投标程序"：投标的具体步骤和程序。

这是招标文件OCR提取的文字，注意可能存在错误：

{text}

根据这个文件，请提取出上述关键信息。

输出格式如下，一定使用英文冒号和空格分隔，否则无法识别。"是否投标"字段的值为"是"或"否"，表示是否是否投标。
报价: 100
招标人信息: xxx博物馆，地址: xxx街道xxx号，电话: 0123456789，电子邮件: abc@abc.com
代理机构信息: xxx有限公司，地址: xxx街道xxx号，电话: 0123456789，电子邮件: abc@abc.com
项目描述: 这是一个关于xxxx的项目，目标是xxxx，期限为xxxx。
资格要求: 投标者需要有xxxx的经验，且公司规模需要在xxxx以上。
投标程序: 首先需要xxxx，然后xxxx，最后将在xxxx公开投标结果。
是否投标: 是
"""

suggest_prompt = """给定一个招标文件的摘要，我需要你判断是否是否投标。

{text}

品艺堂以策略性设计为核心，拥有从策划到设计，再到工程施工、项目管理的全产业链专业能力。并且在长期的实践工作中，形成了一套以“服务设计”为核心理念的独特的工作方法。为客户提供博物馆展陈、企业展厅布展、专题馆、规划馆、景区规划设计、导视标识等“一站式”文化空间解决方案。该项目是否符合品艺堂的业务范围？仅回答是或否。

回答:
"""

select_prompt = """以下是招标项目的序号和标题:

{text}

品艺堂以策略性设计为核心，拥有从策划到设计，再到工程施工、项目管理的全产业链专业能力。并且在长期的实践工作中，形成了一套以“服务设计”为核心理念的独特的工作方法。为客户提供博物馆展陈、企业展厅布展、专题馆、规划馆、景区规划设计、导视标识等“一站式”文化空间解决方案。以上的项目，有哪些明显不符合品艺堂的业务范围？请去掉明显不符合的项目，仅将符合的项目序号写在下面，以逗号分隔。

例如: 0,2,3,5

链接:
"""

gpt35 = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
gpt4 = ChatOpenAI(model="gpt-4", temperature=0)

summarize_chain = (
    {"text": itemgetter("text")}
    | PromptTemplate.from_template(summary_prompt)
    | gpt35
    | StrOutputParser()
)

select_chain = (
    {"text": itemgetter("text")}
    | PromptTemplate.from_template(select_prompt)
    | gpt4
    | StrOutputParser()
)

suggest_chain = (
    {"text": itemgetter("text")}
    | PromptTemplate.from_template(suggest_prompt)
    | gpt4
    | StrOutputParser()
)


def get_urls_by_keyword(keyword: str) -> pd.DataFrame:
    """通过关键字获取所有的招标url

    Args:
        keyword (str): 关键字

    Returns:
        pd.DataFrame: 招标标题和url
    """
    # 初始化浏览器
    driver = webdriver.Chrome()
    driver.get("http://bulletin.sntba.com")
    time.sleep(1)
    # 输入关键字
    driver.find_element(By.CSS_SELECTOR, "#wordkey").send_keys(keyword)
    driver.find_element(
        By.CSS_SELECTOR,
        "body > div.con > div.mian.contents.clear > div.mian_right.fr > div.search > ul:nth-child(2) > li:nth-child(2) > a",
    ).click()
    # 选择最近两天
    driver.find_element(By.LINK_TEXT, "2天内").click()
    # 获取总页数
    driver.switch_to.frame(driver.find_element(By.CSS_SELECTOR, "#iframe"))
    try:
        total_page = int(
            driver.find_element(
                By.CSS_SELECTOR, "body > div.pagination > label:nth-child(1)"
            ).text
        )
    except selenium.common.exceptions.NoSuchElementException:
        total_page = 1
    print("总页数为：", total_page)
    results = []
    for i in tqdm(range(total_page)):
        time.sleep(1)
        trs = driver.find_elements(By.TAG_NAME, "tr")[1:]
        for tr in trs:
            tds = tr.find_elements(By.TAG_NAME, "td")
            a = tds[0].find_element(By.TAG_NAME, "a")
            title = a.get_attribute("title")
            url = a.get_attribute("href").split("'")[1]
            release_date = tds[-2].text
            start_time = tds[-1].get_attribute("id")
            print(title, url, release_date, start_time)
            results.append([title, url, release_date, start_time])
        if i != total_page - 1:
            driver.find_element(By.LINK_TEXT, "下一页").click()
    driver.quit()
    # TODO: use gpt4 to select the most relevant url
    print("开始选择最相关的招标信息")
    indices = select_chain.invoke({"text": "\n".join([f"{i}. {title}" for i, (title, _, _, _) in enumerate(results)])}).strip().split(",")
    results = [results[int(i)] for i in indices]
    return pd.DataFrame(results, columns=["标题", "链接", "发布日期", "开标时间"])


def get_pdf_text(url: str) -> str:
    """通过url获取pdf的文本

    Args:
        url (str): 招标信息url

    Returns:
        str: pdf中的文本
    """
    # 初始化浏览器
    driver = webdriver.Chrome()
    driver.get(url)
    time.sleep(1)
    # 切换到iframe
    driver.switch_to.frame(driver.find_element(By.CSS_SELECTOR, "#iframe"))
    # 等待pdf加载完成
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "page"))
    )
    pages = driver.find_elements(By.CLASS_NAME, "page")
    print("总页数为：", len(pages))
    text = ""
    text_spans = driver.find_element(By.CLASS_NAME, "textLayer").find_elements(
        By.TAG_NAME, "span"
    )
    if len(text_spans) > 10:
        for span in text_spans:
            text += span.text
    else:
        # 全屏截图
        driver.find_element(By.CSS_SELECTOR, "#presentationMode").click()
        for i in range(len(pages)):
            time.sleep(1)
            driver.save_screenshot(f"./.temp/page.png")
            text += ocr(f"./.temp/page.png")
            driver.find_element(By.CSS_SELECTOR, "body").send_keys(Keys.PAGE_DOWN)
    print(text)
    if "频率过高" in text:
        time.sleep(30)
        driver.refresh()
        return get_pdf_text(url)
    return text


def ocr(fp: Union[str, bytes]) -> str:
    """
    对图片进行OCR识别，将图片中的文本内容识别为字符串。

    Args:
        fp (Union[str, bytes]): 图片文件或者文件名

    Returns:
        str: 识别的文本内容
    """

    # 打开图像
    image = Image.open(fp)
    # 使用Tesseract将图像中的文字识别为字符串
    text = pytesseract.image_to_string(image, lang="chi_sim", config="--psm 6 --oem 1")
    return text.replace(" ", "")


def summarize(text: str) -> dict:
    """
    对文本进行摘要提取，将文本中的重要信息提取出来。

    Args:
        text (str): 文本内容

    Returns:
        dict: 摘要信息
    """
    def translate_summary_to_dict(text: str) -> dict:
        pattern = r"(.*?)\s*:\s*(.*?)(?=\n[^:]*:|$)"
        matches = re.findall(pattern, text, re.DOTALL)
        return {key.strip(): value.strip() for key, value in matches}
    print("开始摘要提取")
    summary = summarize_chain.invoke({"text": text}).strip()
    print("开始建议是否投标")
    suggestion = suggest_chain.invoke({"text": summary}).strip()
    print(summary)
    key_info = translate_summary_to_dict(summary)
    key_info["是否投标"] = suggestion == "是"
    print(key_info)
    return key_info


def send_email(subject, body, to, fpath):
    # 邮件服务器设置
    smtp_server_name = 'smtp.office365.com'
    port = 587
    username = 'david_wyb2001@outlook.com'  # 邮箱地址
    password = 'Wyb2001622ms'  # 邮箱密码

    msg = MIMEMultipart()
    msg['From'] = username
    msg['To'] = ', '.join(to) # 收件人，如果有多个人，请用逗号分隔
    msg['Subject'] = subject

    msg.attach(MIMEText(body, "html"))
    part = MIMEBase('application', "octet-stream")
    part.set_payload(open(fpath, "rb").read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment', filename=fpath.split("/")[-1]) 
    msg.attach(part)

    try:
        server = smtplib.SMTP(smtp_server_name, port)
        server.ehlo()
        server.starttls()
        server.login(username, password)
        server.sendmail(username, to, msg.as_string())
        server.close()
        print('邮件发送成功.')
    except Exception as e:
        print(f'邮件发送失败: {e}')

async def async_summary_text(text):
    # 这里我们假设langchain有异步版本的函数ainvoke()
    summary = await summarize_chain.ainvoke({"text": text})
    return json.loads(summary)


def df_to_html(df):
    html = '<html><head><style>body { max-width: 800px; }</style></head><body>\n'
    for index, row in df.iterrows():
        html += '<strong>' + str(row['标题']) + '</strong>\n'
        html += '<ul>\n'
        html += '<li><strong>链接: </strong><a href="' + str(row['链接']) + '">链接</a></li>\n'
        html += '<li><strong>投标价格(万元): </strong>' + str(row['报价']) + '</li>\n'
        html += '<li><strong>招标机构信息: </strong>' + str(row['招标人信息']) + '</li>\n'
        html += '<li><strong>代理机构信息: </strong>' + str(row['代理机构信息']) + '</li>\n'
        html += '<li><strong>项目描述: </strong>' + str(row['项目描述']) + '</li>\n'
        html += '<li><strong>资格要求: </strong>' + str(row['资格要求']) + '</li>\n'
        html += '<li><strong>投标流程: </strong>' + str(row['投标程序']) + '</li>\n'
        html += '<li><strong>发布日期: </strong>' + str(row['发布日期']) + '</li>\n'
        html += '<li><strong>开标时间: </strong>' + str(row['开标时间']) + '</li>\n'
        html += '</ul>'
    return html


def main():
    # 获取所有关键字
    with open("关键字.txt", "r") as f:
        keywords = [line.strip() for line in f.readlines()]

    # 如果已经存在 招标数据YY-mm-dd.csv，直接读取该csv为df，否则创建新df
    file_name = f"招标数据_{dt.now().strftime('%Y-%m-%d')}"
    df = pd.read_csv(f"{file_name}.csv") if os.path.exists(f"{file_name}.csv") else pd.DataFrame(
        columns=[
            "标题",
            "链接",
            "报价",
            "招标人信息",
            "代理机构信息",
            "项目描述",
            "资格要求",
            "投标程序",
            "发布日期",
            "开标时间",
            "是否投标",
            "文本",
        ]
    )

    pbar = tqdm(keywords)
    for keyword in pbar:
        pbar.set_description(f"Processing keyword '{keyword}'")
        new_df = get_urls_by_keyword(keyword)
        df_to_append = new_df[~new_df["链接"].isin(df["链接"])]
        print("新增招标信息数量：", len(df_to_append))
        for _, row in df_to_append.iterrows():
            try:
                text = get_pdf_text(row["链接"])
                summary_result = summarize(text)
                row["报价"] = summary_result.get("报价", "")
                row["招标人信息"] = summary_result.get("招标人信息", "")
                row["代理机构信息"] = summary_result.get("代理机构信息", "")
                row["项目描述"] = summary_result.get("项目描述", "")
                row["资格要求"] = summary_result.get("资格要求", "")
                row["投标程序"] = summary_result.get("投标程序", "")
                row["是否投标"] = summary_result.get("是否投标", "")
                row["文本"] = text
                # df = df.append(row, ignore_index=True)
                df = pd.concat([df, row.to_frame().T], ignore_index=True)
                df.to_csv(f"{file_name}.csv", index=False)
                # drop 是否投标为否的行并保存excel, 并drop是否投标和文本列
                clean_df = df[df["是否投标"] == True].drop(columns=["是否投标", "文本"])
                clean_df.to_excel(f"{file_name}.xlsx", index=False, sheet_name=f"{file_name}")
                with open(f"{file_name}.html", "w") as f:
                    f.write(df_to_html(clean_df))
            except Exception as e:
                print(f"Error processing {row['链接']}: {e}")
            else:
                print(f"Successfully processed {row['链接']}")
    # 发送邮件
    with open(f"{file_name}.html", "r") as f:
        send_email(file_name, f.read(), ["david_wyb2001@outlook.com", "826958975@qq.com", "wtgpyt@126.com"], f"{file_name}.xlsx")
        # send_email(file_name, f.read(), ["david_wyb2001@outlook.com"], f"{file_name}.xlsx")
main()

import http.client
import asyncio
import aiohttp
import aiofiles
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import pdfplumber
import docx
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import json
import os
from goose3 import Goose
from goose3.text import StopWordsChinese
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from plugins.common import settings, allowCROS
import nest_asyncio
from aiohttp import client_exceptions
nest_asyncio.apply()
model_name = settings.librarys.qdrant.model_path
#SERPER_API_KEY=os.getenv("d50d0b2ff04a3bc6ed8101333204d3d0c3281039")
SERPER_API_KEY="d50d0b2ff04a3bc6ed8101333204d3d0c3281039"
g = Goose({'target_language':'zh_cn','browser_user_agent': 'Version/5.1.2 Safari/534.52.7','stopwords_class': StopWordsChinese})
model_kwargs = {'device': settings.librarys.qdrant.device}
encode_kwargs = {'normalize_embeddings': False}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
#async def download_file(url):
#    local_filename = url.split('/')[-1]
#    async with requests.get(url, stream=True) as r:
#        with open(local_filename, 'wb') as f:
#            for chunk in r.iter_content(chunk_size=8192):
#                f.write(chunk)
#    return local_filename
def extract_pdf_content(pdf_filename):
    with pdfplumber.open(pdf_filename) as pdf:
        return '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text())
def extract_docx_content(docx_filename):
    doc = docx.Document(docx_filename)
    return '\n'.join(p.text for p in doc.paragraphs)
async def download_file(session, url):
    local_filename = url.split('/')[-1]
    async with session.get(url) as response:
        if response.status == 200:
            f = await aiofiles.open(local_filename, mode='wb')
            await f.write(await response.read())
            await f.close()
            return local_filename
async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def download_and_extract_file(session, file_link):
    filename = await download_file(session, file_link)
    article_text = ''
    if filename:
        if filename.endswith('.pdf'):
            print(f'PDF内容:')
            article_text = extract_pdf_content(filename)
            print(article_text[0:100])
        elif filename.endswith('.docx'):
            print(f'DOCX内容:')
            article_text = extract_docx_content(filename)
            print(article_text[0:100])
    return article_text
async def parse_article(url):
    async with aiohttp.ClientSession() as session:
        try:
            article_text = ''
            if url.endswith('.pdf') or url.endswith('.docx'):
                #filename = await download_file(session,url)
                article_text = await download_and_extract_file(session, url)
            else:
                html = await fetch(session, url)
                #print(html)
                #g = Goose()
                g = Goose({'target_language':'zh_cn','browser_user_agent': 'Version/5.1.2 Safari/534.52.7','stopwords_class': StopWordsChinese})
                article = g.extract(raw_html=html)
                article_text = article.cleaned_text
                tasks = []
                if len(article_text) < 200:
                    soup = BeautifulSoup(html, 'html.parser')
                    for link in soup.find_all('a', href=True):
                        file_link = link['href']
                        if file_link.endswith('.pdf') or file_link.endswith('.docx'):
                            file_link = urljoin(url, link['href'])
                            print('pdf or word in html')
                            print(file_link)
                            article_text = await download_and_extract_file(session, file_link)
                            if len(article_text) > 200:
                                break
                            #task = asyncio.create_task(download_and_extract_file(session, file_link))
                            #tasks.append(task)
                if tasks:
                    await asyncio.gather(*tasks)

        #except client_exceptions.ClientConnectorError:
        except Exception as e:
            print(f"Connection failed for URL: {url}. Error: {e}")
            # 在这里处理异常，例如记录日志、返回一个默认值等
            return None  # 或者您想返回的任何东西
        #return article.cleaned_text
        return article_text

async def mymain(urls):
    tasks = [parse_article(url) for url in urls]
    #if loop is None:
    #loop = asyncio.get_event_loop()
    #result = loop.run_until_complete(asyncio.gather(*tasks))
    articles = await asyncio.gather(*tasks)
    #for article in articles:
    #    print(article)
    #return result
    return articles
def get_urls_content(urls):
    #loop = asyncio.get_event_loop()
    articles = []
    #if loop.is_running():
    #    articles = loop.create_task(mymain(urls))
    #else:
    #import ipdb
    #ipdb.set_trace()
    #articles = asyncio.run(mymain(urls))
    #urls = ['https://www.ndrc.gov.cn/xxgk/zcfb/ghwb/202103/t20210323_1270124.html']
    #articles = asyncio.run(mymain(urls),debug=True)
    articles = asyncio.run(mymain(urls))
    articles = [e for e in articles if e]
    return articles
def find(search_query,step = 0):
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
    "q": search_query
    })
    headers = {
    'X-API-KEY': SERPER_API_KEY,
    'Content-Type': 'application/json'
    }
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    #import ipdb
    #ipdb.set_trace()
    data = res.read()
    data=json.loads(data)
    clean_data = data['organic']
    content = get_content(clean_data)
    related_res = get_related_content(search_query,content)
    #print(data)
    #l=[{'title': "["+organic["title"]+"]("+organic["link"]+")", 'content':organic["snippet"]} for organic in data['organic']]
    #try:
    #    if data.get("answerBox"):
    #        answer_box = data.get("answerBox", {})
    #        l.insert(0,{'title': "[answer："+answer_box["title"]+"]("+answer_box["link"]+")", 'content':answer_box["snippet"]})
    #except:
    #    pass
    #return l,data
    #return content
    return related_res
def get_content(res_li):
    """
    {'title': '《南阳市“十四五”现代服务业发展规划》政策解读', 'link': 'http://henan
    .kjzch.com/nanyang/2022-06-17/819566.html', 'snippet': '顺应产业融合需求，秉承“
    两业并举”，以服务产业升级、提高流通效率为导向，大力发展现代金融、现代物流、科技
    服务、信息服务、中介服务、节能环保服务、 ...', 'date': 'Jun 17, 2022', 'attribu
    tes': {'南阳市人民政府办公室': '2022-06-17'}, 'position': 1}
    """
    res = []
    len_str = 0
    urls = []
    for da in res_li:
        link = da['link']
        urls.append(link)
        #try:
        #    article = g.extract(url=link)
        #    title = article.title
        #    cleaned_text = article.cleaned_text
        #    len_str += len(title)
        #    len_str += len(cleaned_text)
        #    res.append(title)
        #    res.append(cleaned_text)
        #except Exception as e:
        #    print(e)
        ##if len_str > 1500:
        #if len_str > 12000:
        #    break
    #import ipdb
    #ipdb.set_trace()
    res = get_urls_content(urls)
    res =  '\n'.join(res)
    return res

def get_related_content(query,content):
    def clean_text(content):
        res = []
        for txt in content.split('\n'):
            if not txt:
                continue
            if len(txt) < 5:
                continue
            res.append(txt.strip())
        return '\n'.join(res)

    #content_li = content.split('。')
    #import ipdb
    #ipdb.set_trace()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=25,separators=["\n\n", "\n","。","\r","\u3000"])
    #texts = text_splitter.split_documents(content)
    content = clean_text(content)
    content_li = text_splitter.split_text(content)
    content_li.append(query)
    embedding = hf_embeddings.embed_documents(content_li)
    score = cosine_similarity([embedding[-1]],embedding[0:-1])
    idxs = score.argsort()
    idxs = idxs[0][::-1]
    res = ['']*len(content_li)
    len_str = 0
    #import ipdb
    #ipdb.set_trace()
    for idx in idxs:
        s = score[0][idx]
        if s < 0.75:
            continue
        sub_content = content_li[idx]
        if not sub_content:
            continue
        if len(sub_content) < 15:
            continue
        len_str += len(sub_content)
        res[idx]=sub_content
        #if len_str > 1000:
        #if len_str > 1500:
        if len_str > 1500:
        #if len_str > 8500:
            break
    final_res = []
    #len_res = len(res)
    for idx,txt in enumerate(res):
        #start = idx - 3 if idx - 3 >= 0 else 0
        #end = idx + 3 if idx + 3 < len_res else len_res - 1
        #useful_count = len([i for i in res[start:end+1] if i])
        #ratio = useful_count / len(res[start:end+1])
        #if ratio > 0.28:
        #    final_res.append(content_li[idx])
        if txt.strip() and len(txt) > 10:
            final_res.append(txt)

    res = '\n\n'.join(final_res)
    #res = '\n'.join(res)
    #return res[0:1700]
    return res[0:2500]
    #return res[0:8700]



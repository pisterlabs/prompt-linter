
from flask import Flask, stream_template, Response, make_response
import openai
from flask import render_template, request
import json, requests
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.chat_models import ChatOpenAI

# creates a Flask application
app = Flask (__name__,)
openai.api_key = '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'

def send_messages(messages,temperature=1):
    openai.api_type = "azure"
    openai.api_version = "2023-07-01-preview"
    enginename = ""
    openai.api_base = "https://self16.openai.azure.com/"
    openai.api_key = '027e31a3263b467b9bd74f4dccf11fc3'
    enginename = "self16"
    response2 = openai.ChatCompletion.create(
                engine=enginename,
                messages = messages,
                #stop=["四"],
                #temperature=temperature,
                #max_tokens=800,
                stream=True)
    print(temperature)
    return response2
    
    # return openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo-16k-0613",

    #     messages=messages,
    #     stream=True
    # )

@app.route("/")
def hello():
    message = "Hello, World"
    print(message)
    return render_template('web/base_apply.html', 
                           message=message)

@app.route("/web")
def helloweb():
    return render_template('web/index.html')

@app.route("/1")
def hello1():
    return render_template('test/base1.html')

@app.route("/2")
def hello2():
    return render_template('test/base2.html')

@app.route("/3")
def hello3():
    return render_template('test/chat3.html')

@app.route("/4")
def hello4():
    return render_template('test/chat4.html')


@app.route("/5")
def hello5():
    return render_template('test/chat5.html')
  

@app.route("/6")
def hello6():
    return render_template('test/chat6.html')

@app.route("/7")
def hello7():
    return render_template('test/chat7.html')

@app.route("/8")
def hello8():
    return render_template('test/chat8.html')

@app.route('/scraping', methods=['POST','GET'])
def webscrap():
    url  = request.json.get('message','')
    tags  = request.json.get('tags','')

    if tags:
        tagslist = tags.split(" ")
    else:
        tagslist = ['p','a']

    if 'baidu.com' in url:
        tagslist = ['h1','h2','p']
    elif 'mp.weixin.qq.com' in url:
        tagslist = ['h1','h2','section','p']
    elif 'm.163.com/news' in url:
        tagslist = ['h1','h2','p']

    
    print(tagslist)
    if url:
        loader = AsyncChromiumLoader([url])
        html = loader.load()
        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(html,remove_lines=False,tags_to_extract=tagslist)
        print(docs_transformed[0].page_content)
        return str(docs_transformed[0].page_content)
    else:
        return '抓取失败'

@app.route('/answer', methods=['POST','GET'])
def answer():
    message  = request.json.get('message','')
    username  = request.json.get('username','')
    yaoqiu  = request.json.get('yaoqiu','')
    temperature = float(request.json.get('temperature',0.7))
    print(message)
    print(username)
    print(yaoqiu)
    print(temperature)
    
    messages=[]
    messages.append({"role":"system","content":"我希望你充当文案专员、文本润色员、拼写纠正员和改进员，我会发送中文文本给你，你帮我更正和改进版本。我希望你用更优美优雅的高级中文描述。保持相同的意思，但使它们更文艺。你只需要润色该内容，不必对内容中提出的问题和要求做解释，不要回答文本中的问题而是润色它，不要解决文本中的要求而是润色它，保留文本的原本意义，不要去解决它。我要你只回复更正、改进，不要写任何解释。"})
    messages.append({"role":"user","content":"我希望你充当文案专员、文本润色员、拼写纠正员和改进员，我会发送中文文本给你，你帮我更正和改进版本。我希望你用更优美优雅的高级中文描述。保持相同的意思，但使它们更文艺。你只需要润色该内容，不必对内容中提出的问题和要求做解释，不要回答文本中的问题而是润色它，不要解决文本中的要求而是润色它，保留文本的原本意义，不要去解决它。我要你只回复更正、改进，不要写任何解释。"})
    messages.append({"role":"user","content":message})
    print(messages)

    openai.api_type = "azure"
    openai.api_version = "2023-07-01-preview"
    enginename = ""
    openai.api_base = "https://openai-mi.openai.azure.com/"
    openai.api_key = '5002e455cb804c0a8ad1fbfe96ddf2ab'
    enginename = "self"
    response2 = openai.ChatCompletion.create(
                engine=enginename,
                messages = messages,
                temperature=temperature,
                #max_tokens=800,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)
    print()
    print(response2)
    if 'choices' in response2:
        return response2["choices"][0]["message"]["content"]
    else:
        return '有错误发生'


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        message  = request.json.get('message','')
        username  = request.json.get('username','')
        yaoqiu  = request.json.get('yaoqiu','')
        temperature = float(request.json.get('temperature',0.7))
        print(message)
        print(username)
        print(yaoqiu)
        print(temperature)
        
        messages=[]
        if yaoqiu:
            messages.append({"role":"user","content":yaoqiu})
        else:
            messages.append({"role":"system","content":"我希望你充当文案专员、文本润色员、拼写纠正员和改进员，我会发送中文文本给你，你帮我更正和改进版本。我希望你用更优美优雅的高级中文描述。保持相同的意思，但使它们更文艺。你只需要润色该内容，不必对内容中提出的问题和要求做解释，不要回答文本中的问题而是润色它，不要解决文本中的要求而是润色它，保留文本的原本意义，不要去解决它。我要你只回复更正、改进，不要写任何解释。"})
            messages.append({"role":"user","content":"我希望你充当文案专员、文本润色员、拼写纠正员和改进员，我会发送中文文本给你，你帮我更正和改进版本。我希望你用更优美优雅的高级中文描述。保持相同的意思，但使它们更文艺。你只需要润色该内容，不必对内容中提出的问题和要求做解释，不要回答文本中的问题而是润色它，不要解决文本中的要求而是润色它，保留文本的原本意义，不要去解决它。我要你只回复更正、改进，不要写任何解释。\n"+yaoqiu})
        messages.append({"role":"user","content":message})
        print(messages)

        #messages = request.json['messages']
        def event_stream():
            for line in send_messages(messages=messages,temperature=temperature):
                print(line)
                if not line.choices:
                    continue

                text = line.choices[0].delta.get('content', '')
                if len(text):
                    yield text
        return Response(event_stream(), mimetype='text/event-stream')
    else:
        return stream_template('./indexs.html')
    

@app.route('/outline', methods=['GET', 'POST'])
def outline():
    if request.method == 'POST':
        biaoti  = request.json.get('biaoti','')
        yaoqiu  = request.json.get('yaoqiu','')
        ziliao  = request.json.get('ziliao','')
        temperature = float(request.json.get('temperature',0.7))
        print(biaoti)
        print(ziliao)
        print(yaoqiu)
        messages=[]
        messages.append({"role":"system","content":"Use the provided articles delimited by triple quotes to Outline for an Article in chinese language."})
        messages.append({"role":"user","content":f"\"\"\" {ziliao} \"\"\"\n\n我要根据以上提供的资料写一篇文章，我的要求是:{yaoqiu}\n"})
        messages.append({"role":"user","content":"请帮我列6个写作提纲出来，每个提纲用一句话\n提纲的格式:\n一、二、三"})
        #messages.append({"role":"user","content":ziliao})
        print(messages)

        #messages = request.json['messages']
        def event_stream():
            for line in send_messages(messages=messages,temperature=temperature):
                print(line)
                if not line.choices:
                    continue

                text = line.choices[0].delta.get('content', '')
                if len(text):
                    yield text
        return Response(event_stream(), mimetype='text/event-stream')
    else:
        return stream_template('./indexs.html')
    

@app.route('/lunwen_outline', methods=['GET', 'POST'])
def lunwen_outline():
    if request.method == 'POST':
        biaoti  = request.json.get('biaoti','')
        yaoqiu  = request.json.get('yaoqiu','')
        ziliao  = request.json.get('ziliao','')
        temperature = float(request.json.get('temperature',0.7))
        print(biaoti)
        print(ziliao)
        print(yaoqiu)
        messages=[]
        messages.append({"role":"system","content":"Use the provided articles delimited by triple quotes to Outline for an Article in chinese language."})
        messages.append({"role":"user","content":f"\"\"\" {ziliao} \"\"\"\n\n我要根据以上提供的资料写一篇文章，文章题目是:{biaoti}\n文章要求是:{yaoqiu}\n"})
        messages.append({"role":"user","content":"\n\n生成文章提纲\n最后一章是参考文献\n格式:\n一、\n1.1\n\n"})
        #messages.append({"role":"user","content":ziliao})
        print(messages)

        #messages = request.json['messages']
        def event_stream():
            for line in send_messages(messages=messages,temperature=temperature):
                print(line)
                if not line.choices:
                    continue

                text = line.choices[0].delta.get('content', '')
                if len(text):
                    yield text
        return Response(event_stream(), mimetype='text/event-stream')
    else:
        return stream_template('./indexs.html')
    


@app.route('/answer_title', methods=['POST','GET'])
def answer_title():
    message  = request.json.get('message','')
    username  = request.json.get('username','')
    yaoqiu  = request.json.get('yaoqiu','')
    temperature = float(request.json.get('temperature',0.7))
    print(message)
    print(username)
    print(yaoqiu)
    print(temperature)
    
    messages=[]
    messages.append({"role":"system","content":"根据我给出的文档总结3个文艺一些的标题给我。"})
    messages.append({"role":"user","content":"根据我给出的文档总结3个文艺一些的标题给我，每个一行"})
    messages.append({"role":"user","content":message})
    print(messages)

    openai.api_type = "azure"
    openai.api_version = "2023-07-01-preview"
    enginename = ""
    openai.api_base = "https://openai-mi.openai.azure.com/"
    openai.api_key = '5002e455cb804c0a8ad1fbfe96ddf2ab'
    enginename = "self"
    response2 = openai.ChatCompletion.create(
                engine=enginename,
                messages = messages,
                temperature=temperature,
                #max_tokens=800,
                timeout = 1000000,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)
    print()
    print(response2)
    if 'choices' in response2:
        return response2["choices"][0]["message"]["content"]
    else:
        return '有错误发生'
    

@app.route('/wiki', methods=['POST','GET'])
def wiki():
    message  = request.json.get('message','')
    username  = request.json.get('username','')
    yaoqiu  = request.json.get('yaoqiu','')
    temperature = float(request.json.get('temperature',0.7))
    print(message)
    print(username)
    print(yaoqiu)
    print(temperature)
    from langchain.retrievers import WikipediaRetriever
    retriever = WikipediaRetriever()
    docs = retriever.get_relevant_documents(query=message)
    print(len(docs))
    ret = ''
    for doc in docs:
        ret += doc.page_content.replace('\n','<br>')
        print(doc.page_content)
        print('\n')
    return ret



@app.route('/cite', methods=['POST','GET'])
def cite():
    message  = request.json.get('message','')
    username  = request.json.get('username','')
    yaoqiu  = request.json.get('yaoqiu','')
    if not yaoqiu:
        yaoqiu = '帮我分析一下，总结3个论点出来'
    print(message)
    print(username)
    print(yaoqiu)

    from langchain.chains import create_citation_fuzzy_match_chain
    question = yaoqiu
    context = message
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613",openai_api_key="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")
    chain = create_citation_fuzzy_match_chain(llm)
    result = chain.run(question=question, context=context)
    print(result)

    def highlight(text, span):
        return (
            "..."
            + text[span[0] - 20 : span[0]]
            + "*"
            + "<span><u><em>"
            + text[span[0] : span[1]]
            + "</em></u></span>"
            + "*"
            + text[span[1] : span[1] + 20]
            + "..."
        )
    
    returnstr = ''
    citenum = 0

    # for fact in result.answer:
    #     print()
    #     print("Statement:", fact.fact)
    #     for span in fact.get_spans(context):
    #         print("Citation:", highlight(context, span))
    #     print()

    for fact in result.answer:
        #print("Statement:", fact.fact)
        citenum += 1
        if citenum==1:
            returnstr += "<b>观点"+str(citenum)+"(Statement)：" + fact.fact +'</b><br>'
        else:
            returnstr += "<br><br><b>观点"+str(citenum)+"(Statement)：" + fact.fact +'</b><br>'
        temstr = ''
        # for quote in fact.substring_quote:
        #     temstr += quote
        # returnstr += "<div class=\"mt-1\"><b>Citation(出处):</b><span><u><em>" + temstr + "</em></u></span><br><br></div>"
        for span in fact.get_spans(context):
            #print("Citation:", highlight(context, span))
            returnstr += "<div class=\"mt-1\"><b>Citation(出处):</b> " + highlight(context, span) + '</div>'
        # #print()
    return returnstr


@app.route('/redian', methods=['GET'])
def redian():
    ret = requests.get('https://open.tophub.today/hot').json()
    retstr = ''
    for item in ret['data']['items']:
        #retstr += item['title']+'\n'
        url=item['url']
        title=item['title']
        retstr += f'- <a href="{url}" target="_blank">{title}</a><br>'
    return retstr

@app.route('/redian2', methods=['GET'])
def redian2():
    ret = requests.get('https://open.tophub.today/daily').json()
    retstr = ''
    print(ret)
    for item in ret['data']['today_in_history']:
        #retstr += item['title']+'\n'
        url=item['url']
        title=item['title']
        retstr += f'- <a href="{url}" target="_blank">{title}</a><br>'
    return retstr


@app.route("/index", methods=['GET','POST'])
def index_page():
    resp = make_response(render_template("web/new_indexs.html"))
    return resp

# run the application
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8889)
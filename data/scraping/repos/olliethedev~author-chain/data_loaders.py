from gnews import GNews
from json import dumps
from langchain import OpenAI, GoogleSearchAPIWrapper
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.agents import initialize_agent, Tool
from langchain.docstore.document import Document

def get_input_articles(topic):
    google_news = GNews(language='en', max_results=10)
    search_date_filter = 'when:7d'
    json_resp = google_news.get_news(f'{topic}  {search_date_filter}')

    articles = []
    for item in json_resp:
        try :
            article = google_news.get_full_article(
                item['url'])
            meta = {
            'title': article.title,
            'text': article.text,
            'url': item['url'],
            'images' : article.images,
            }
            articles.append(meta)
            print(dumps(meta))
        except Exception as e:
            print(e)

    print(f'source articles length: {len(articles)}')

    llm = OpenAI(temperature=0, model_name='text-davinci-003')

    text_splitter = CharacterTextSplitter()

    for article in articles:
        texts = text_splitter.split_text(article['text'])
        docs = [Document(page_content=t) for t in texts]
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)
        article['summary'] = summary
        print('Summary:'+ summary)
    
    return articles

def get_questions_answers(questions):
    extra_data= ''

    for question in questions.split('\n'):
        if(len(question) == 0):
            continue
        print('Researching question: '+question)
        try: 
            answer = get_extra_data(question)
            extra_data += f'{question}\n{answer}\n'
            print(extra_data)
        except Exception as e:
            print(e)
    
    
    return extra_data

def get_extra_data(question):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    search = GoogleSearchAPIWrapper(k=5)

    tools = [
        Tool(
            name = "Current Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world"
        ),
    ]

    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

    search_chain = initialize_agent(tools, llm, agent="chat-zero-shot-react-description", verbose=True, memory=memory, max_iterations=10)

    return search_chain.run(input=f'Answer the following question, use search once: \n{question}')

from typing import List
import asyncio
from concurrent.futures import ProcessPoolExecutor
import datetime
import pynecone as pc
import requests
from bs4 import BeautifulSoup
import pandas as pd
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage

from langchain.utilities import DuckDuckGoSearchAPIWrapper
import tiktoken

import os

key = open('../api-key', 'r').readline()
os.environ["OPENAI_API_KEY"] = key



###########################################################
# Helpers
def build_summarizer(llm):
    system_message = "assistant는 user의 내용을 bullet point 3줄로 요약하라. 영어인 경우 한국어로 번역해서 요약하라."
    system_message_prompt = SystemMessage(content=system_message)

    human_template = "{text}\n---\n위 내용을 bullet point로 3줄로 한국어로 요약해"
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,
                                                    human_message_prompt])

    chain = LLMChain(llm=llm, prompt=chat_prompt)
    return chain


def truncate_text(text, max_tokens=3000):
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:  # 토큰 수가 이미 3000 이하라면 전체 텍스트 반환
        return text
    return enc.decode(tokens[:max_tokens])


def clean_html(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ' '.join(soup.stripped_strings)
    return text


def task(search_result):
    title = search_result['title']
    url = search_result['link']
    snippet = search_result['snippet']

    content = clean_html(url)
    full_content = f"제목: {title}\n발췌: {snippet}\n전문: {content}"

    full_content_truncated = truncate_text(full_content, max_tokens=3500)

    summary = summarizer.run(text=full_content_truncated)

    result = {"title": title,
              "url": url,
              "content": content,
              "summary": summary
              }

    return result


###########################################################
# Instances
llm = ChatOpenAI(temperature=0.8)

search = DuckDuckGoSearchAPIWrapper()
search.region = 'kr-kr'

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

summarizer = build_summarizer(llm)


###########################################################
# Web

class Data(pc.Model, table=True):
    """A table for questions and answers in the database."""

    title: str
    content: str
    url: str
    summary: str
    timestamp: datetime.datetime = datetime.datetime.now()


class State(pc.State):
    """The app state."""

    is_working: bool = False
    columns: List[str] = ["title", "url", "summary"]
    topic: str = ""

    async def handle_submit(self):
        self.is_working = True
        yield

        topic = self.topic

        #검색 결과를 3개 가져옴
        search_results = search.results(topic, num_results=3)

        #병렬 실행
        with ProcessPoolExecutor() as executor:
            with pc.session() as session:

                for s in search_results:
                    # {"title":~~~, "snippet":~~, "link":~~~}
                    s = await asyncio.get_running_loop().run_in_executor(executor, task, s)
                    record = Data(title=s['title'],
                                  content=s['content'],
                                  url=s['url'],
                                  summary=s['summary'])
                    #pynecone db에 저장
                    session.add(record)
                    session.commit()
                    yield

        self.is_working = False

    @pc.var
    def data(self) -> List[Data]:
        """Get the saved questions and answers from the database."""
        with pc.session() as session:
            samples = (
                session.query(Data)
                .order_by(Data.timestamp.asc())
                .all()
            )
            return [[s.title, s.url, s.summary] for s in samples]

    def export(self):
        with pc.session() as session:
            samples = (
                session.query(Data)
                .all()
            )
            d = [{"title": s.title,
                  "url": s.url,
                  "summary": s.summary,
                  "content": s.content} for s in samples]

            df = pd.DataFrame(d)
            df.to_excel("./exported.xlsx")

    def delete_all(self):
        with pc.session() as session:
            samples = (
                session.query(Data)
                .all()
            )

            for s in samples:
                session.delete(s)
            session.commit()


def index() -> pc.Component:
    return pc.center(
        pc.vstack(
            pc.heading("뉴스 크롤링 & 요약 서비스", font_size="2em"),
            pc.input(placeholder="topic", on_blur=State.set_topic),
            pc.hstack(
                pc.button("시작", on_click=State.handle_submit),
                pc.button("excel로 export", on_click=State.export),
                pc.button("모두 삭제", on_click=State.delete_all),
            ),
            pc.cond(State.is_working,
                    pc.spinner(
                        color="lightgreen",
                        thickness=5,
                        speed="1.5s",
                        size="xl",
                    ),),
            pc.data_table(
                data=State.data,
                columns=State.columns,
                pagination=True,
                search=True,
                sort=False,
            ),
            width="80%",
            font_size="1em",
        ),
        padding_top="10%",
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index)
app.compile()

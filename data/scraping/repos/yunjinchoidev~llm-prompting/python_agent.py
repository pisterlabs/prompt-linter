from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain.agents.agent_toolkits import create_python_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import PythonREPLTool

load_dotenv()

# app = FastAPI()
templates = Jinja2Templates(
    directory="templates"
)  # Assuming you have templates in a folder called "templates"


class TextData(BaseModel):
    text: str


# @app.post("/wordcloud/")
def create_word_cloud(text_data: TextData):
    python_agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k-0613"),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    result = python_agent_executor.run(
        f"make word cloud from the text {text_data.text} by using python."
    )

    # Depending on the output of your executor, you might need to adjust this.
    return {"result": result}


if __name__ == "__main__":
    import uvicorn

    # uvicorn.run(app, host="0.0.0.0", port=8000)

    txt = """
    [앵커]
원자폭탄 개발자의 고뇌와 모순을 그린 크리스토퍼 놀런 감독의 '오펜하이머'가 글로벌 흥행 속에 한국 관객들을 만납니다.

이미 놀런 감독의 최대 흥행작에 오른 '오펜하이머'가 국내에서도 압도적인 예매율을 기록해, 한국 영화 기대작들과 한판 승부를 벌입니다.

홍상희 기자입니다.

    """

    #
    # create_word_cloud(
    #     TextData(text=txt)
    # )

    python_agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k-0613"),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    result = python_agent_executor.run(
        f"""
        summarize text "원자폭탄 개발자의 고뇌와 모순을 그린 크리스토퍼 놀런 감독의 '오펜하이머'가 글로벌 흥행 속에 한국 관객들을 만납니다."
        make word cloud from the text by using python.
        and save img file as 'wordcloud.png'
        """
    )

    # result = python_agent_executor.run(
    #     f"""
    #         Please develop a fastapi that displays "hello world!"
    #     """
    # )

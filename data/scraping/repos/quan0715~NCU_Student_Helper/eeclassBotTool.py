import time
from langchain.tools import BaseTool
from NotionBot import *
from NotionBot.base.Database import *
from .eeclass_notion_db_crawler.eeclass_notion_db_crawler import EEClassNotionDBCrawler

from langchain.memory import ConversationBufferMemory
from typing import Any, Coroutine, Optional, Type
from uuid import UUID
from pydantic import BaseModel, Field
import os
from datetime import date, timedelta, timezone
from dotenv import load_dotenv
import enum, random
load_dotenv()
AUTH=os.getenv("NOTION_AUTH")
PAGE_ID=os.getenv("PAGE_ID")

dbc = EEClassNotionDBCrawler(
    auth=AUTH,
    page_id=PAGE_ID
)
course = dbc.get_all_courses()
homework = dbc.get_homework()

CourseName = enum.Enum('a', {'a'+str(random.randint(0,10000000)): c for c in course})
HomeworkName = enum.Enum('a', {'a'+str(random.randint(0,10000000)): c for c in homework})

def set_exit_state() -> None:
    from .chatBotModel import default_message
    from .chatBotExtension import jump_to
    jump_to(default_message, True)
    get_agent_pool_instance().remove()

class ExitTool(BaseTool):
    name = "ExitTool"
    description = """
    Useful in the situations below:
    - The user has no high intention to know things about his bulletin, homework, course and course material information. For example, "quit", "exit", "I don't want to talk about it."
    - The user want to break or exit this conversation.
    - The topic of the conversation isn't highly related to college course query.
    """

    def _run(self, *args: Any, **kwargs: Any) -> str:
        set_exit_state()
        return "The process exited successfully."


class HomeworkTitleInput(BaseModel):
    """Input for homework title"""
    course_title: str = Field(
        ...,
        description=f"It's a college course that the user takes in this semester. If user doesn't offer this, please return empty string."
    )
    homework_title_input: str = Field(
        ...,
        description="It's a homework title that is assigned in a college course."
    )


class SpecificCourseName(BaseModel):
    course_name: CourseName = Field(description=f"This course name must be one of the {course}. It must be completely the same string.")

class SpecificHomeworkName(BaseModel):
    homework_title: HomeworkName = Field(description=f"This homework name must be one of the {homework}. It must be completely the same string.")

class HomeworkAlertInput(BaseModel):
    """Input for homework alert"""
    days_left: int = Field(
        ...,
        description=f"Use to find the homeworks that is due between today - days_left days and today. Date format shoud be'YYYYMMDDTHHMMSS'. Current time is {date.today()}"
    )

class SearchNearestCourseTitle(BaseTool):
    name = "search_nearest_course_title"
    description = "這是一個EECLASS搜尋. 給定一個課程名稱，回傳與課程列表中最接近的一個字串"
    @staticmethod
    def get_course_full_name(course_name: str):
        return course_name
    
    def _run(self, course_name: str):
        result = SearchNearestCourseTitle.get_course_full_name(course_name)
        return result
    
    args_schema: Optional[Type[BaseModel]] = SpecificCourseName

class HomeworkContent(BaseTool):
    name="Homework_content_recommendation"
    description="這是一個EECLASS搜尋. Please give an idea from the homework content and summarize all the detail."

    @staticmethod
    def make_recommendation(hw_title: str):
        for hw in dbc.get_homework():
            if hw.title == hw_title:
                return hw
    def _run(self, hw_title: str) -> Any:
        result = HomeworkContent.make_recommendation(hw_title)
        return result
    
    args_schema: Optional[Type[BaseModel]] = SpecificHomeworkName

class CoursetoHomework(BaseTool):
    name = "Use_course_name_to_fetch_homework"
    description = "這是一個EECLASS搜尋. User will input a course name, and please return all the homework in that course."
        
    @staticmethod
    def course_to_hw(course_name: str):
        dbc = EEClassNotionDBCrawler(
            auth=AUTH,
            page_id=PAGE_ID
        )
        homework_list = dbc.get_homework()
        filtered_homework_list = []
        for hw in homework_list:
            if hw.course == course_name.value:
                filtered_homework_list.append(dict(
                    title=hw.title,
                    homework_type=hw.homework_type,
                    deadline=hw.deadline,
                    content=hw.content
                ))
        return filtered_homework_list

    def _run(self, course_name: str):
        result = CoursetoHomework.course_to_hw(course_name)
        return result
    
    args_schema: Optional[Type[BaseModel]] = SpecificCourseName

class BulletinRetrieve(BaseTool):
    name="search_all_bulletin"
    description=f"這是一個EECLASS搜尋. 請條列式地將所有公告列出來. By the way, current time is {date.today()}"
    @staticmethod
    def get_bulletin_db():
        bulletin_list = []
        for bu in dbc.get_bulletin()[:10]:
            bulletin_list.append(dict(
                title=bu.title,
                content=bu.content,
                announce_date=bu.announce_date,
                course=bu.course
            ))
        return bulletin_list
    def _run(self, *argc, **kargs):
        result = BulletinRetrieve.get_bulletin_db()
        return result


class HomeworkRetrieve(BaseTool):
    # name = "Homework_Content_Recommendation_system"
    # description = f"User will give only homework name or course name with homework name. Please make some detail recommendation to each homework content. Or give some useful idea on each content. Or what it is about. You can summarize it. By the way, current time is {date.today()}"
    name = "search_all_homework"
    description = f"這是一個EECLASS搜尋. 請幫忙搜尋所有課程相關的作業，並回傳搜尋結果. By the way, current time is {date.today()}"
    @staticmethod
    def get_homework_db():
        homeworks = []
        for hw in dbc.get_homework()[:10]:
            homeworks.append(dict(
                    title=hw.title,
                    homework_type=hw.homework_type,
                    deadline=hw.deadline,
                    content=hw.content
                ))
        return homeworks

    def _run(self, *args, **kargs):
        result = self.get_homework_db()
        return result
    
class HomeworkAlertTool(BaseTool):
    name = "Homework_alert_submission_system"
    description = "這是一個EECLASS搜尋. Check whether there are any homeworks that is close to end date but not submitted."

    @staticmethod
    def get_alert_homework(days_left: int=1000) -> List[str]:
        if days_left == 1000:
            return ["我也不知道誒"]
        load_dotenv()
        auth = os.getenv("NOTION_AUTH")
        notion_bot = Notion(auth)
        homework_db: Database = notion_bot.get_database(os.getenv("HOMEWORK_DB"))
        now = datetime.strptime(datetime.now(tz=timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M"), "%Y-%m-%d %H:%M")
        alert_list = []
        for h in homework_db.query():
            end_date = datetime(*time.strptime(h['properties']['Deadline']['date']['end'], "%Y-%m-%dT%H:%M:%S.000+00:00")[:6])
            submission_status = h['properties']['Status']['select']['name']
            course = h['properties']['Course']['select']['name']
            homework_title = h['properties']['Title']['title'][0]['plain_text']
            if submission_status == "未完成" and now <= end_date <= now+timedelta(days=days_left):
                alert_list.append(f"課程: {course}\n作業: {homework_title}\n剩餘時間: {str(end_date-now)}\n")
        return alert_list

    def _run(self, days_left: int):
        result = self.get_alert_homework(days_left)
        return "--------split--------\n".join(result)
    
    args_schema: Optional[Type[BaseModel]] = HomeworkAlertInput
        
    # open_ai_agent.run("請問10天內有作業要交嗎?")
    # open_ai_agent.run("請問什麼是作業?")
    # open_ai_agent.run("請問Lab 1是在幹嘛?")
    # open_ai_agent.run("請問軟體工程實務的Final Project是在幹嘛?")
    # open_ai_agent.run("請問軟工有什麼作業嗎?")
    # open_ai_agent.run("我這週有什麼上班嗎")
    # open_ai_agent.run("請問Lab 1")
    # open_ai_agent.run("請問總共有哪些公告?")
    # open_ai_agent.run("請問最近有哪些新公告?")
    # open_ai_agent.run("請問跟Linux最有關的是哪一門課?")
from .langChainAgent import LangChainAgent

def getEETool(user_id: str):
    class eeAgentPool(BaseTool):
        name = "EECLASS_query_system"
        description = "Useful to get EECLASS data by using "

        def _run(self) -> Any:
            request = requests.get(
                f"https://api.squidspirit.com/eeclass_api/get_data?user_id=${user_id}"
            )
            if request.status_code != 200:
                return request.json()
            set_exit_state(user_id)
            return f"The notion_token is {request.json()['data']['notion_token']}, and notion_template_id is {request.json()['data']['notion_template_id']}. Please remember it."

        def _arun(self, *args: Any, **kwargs: Any) -> Coroutine[Any, Any, Any]:
            raise Exception()
    return eeAgentPool()

class eeAgentPool:
    def __init__(self) -> None:
        self.pool: dict[str, LangChainAgent] = {}
        self.sessions: dict[str, UUID] = {}

    def get(self, user_id: str) -> LangChainAgent | None:
        return self.pool.get(user_id)

    def add(self, user_id: str) -> LangChainAgent:
        agent = LangChainAgent(
            tools=[
                    getEETool(user_id),
                    HomeworkRetrieve(),
                    BulletinRetrieve(),
                    CoursetoHomework(),
                    HomeworkContent(),
                    SearchNearestCourseTitle(),
                    HomeworkAlertTool(),
                    ExitTool()
                ],
                memory=ConversationBufferMemory(
                memory_key="ee", return_messages=True),
            timeout=-1
        )
        return agent

    def remove(self, user_id: str) -> None:
        self.pool.pop(user_id)


__ee_agent_pool_instance = eeAgentPool()
def get_agent_pool_instance():
    return __ee_agent_pool_instance
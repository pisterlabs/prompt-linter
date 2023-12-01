from langchain.tools import BaseTool
from NotionBot import *
from NotionBot.base.Database import *

from typing import Optional, Type
from pydantic import BaseModel, Field
import datetime
import os
import time
from datetime import datetime, timezone, timedelta, date
from dotenv import load_dotenv

class HomeworkAlertInput(BaseModel):
    """Input for homework alert"""
    days_left: int = Field(
        ...,
        description=f"Use to find the homeworks that is due between today - days_left days and today. Date format shoud be'YYYYMMDDTHHMMSS'. Current time is {date.today()}"
    )

class HomeworkAlertTool(BaseTool):
    name = "Homework_alert_submission_system"
    description = "Check whether there are any homeworks that is close to end date but not submitted."

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


if __name__ == "__main__":
    print("test")
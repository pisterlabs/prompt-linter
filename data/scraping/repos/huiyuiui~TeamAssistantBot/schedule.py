from ast import literal_eval
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
from datetime import datetime
from urllib import parse
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from imgurpython import ImgurClient
import matplotlib as mpl
mpl.rcParams['font.sans-serif']='Arial Unicode MS'
mpl.rcParams['axes.unicode_minus']=False

class SubtaskInput(BaseModel):
    """Input for Subtasks of Main Task"""
    task: str = Field(
        ...,
        description="Task description for this subtask"
    )
    people: str = Field(
        ...,
        description="The number of people to complete this task"
    )
    startTime: str = Field(
        ...,
        description="Start time of this task"
    )
    endTime: str = Field(
        ...,
        description="End time of this task"
    )
    
class ScheduleGenerateInput(BaseModel):
    """Input for Schedule Generate."""
    main_task: str = Field(
        ...,
        description="Main task symbol for Schedule"
    )
    deadline: str = Field(
        ...,
        description="Time limit for complete all Main tasks and Subtasks"
    )
    subtasks: List[SubtaskInput] = Field(
        ...,
        description="Subtasks list for the main task"
    )
    
class ScheduleTool(BaseTool):
    name = "create_task_schedule"
    description =f"""
    Generate Task Schedule from text.
    According time limit and the number of people to schedule and distribute tasks.
    Start time and end time format should be like 'MM-DD'.
    Start time and end time can't be the same.
    Current time {datetime.now()}.
    Output format should contain image url.
    """

    def _run(self, main_task: str, deadline: str, subtasks: List[SubtaskInput]):
        # print(main_task)
        print(deadline)
        print(subtasks)
        output = {
            "Main_task": main_task,
            "Subtasks": [],
            "ImgUrl" : None
        }

        # Ensure subtasks is a list of dictionaries
        if isinstance(subtasks, dict):
            subtasks = [subtasks]

        for subtask in subtasks:
            subtask_info = {
                "Task": subtask['task'],
                "People": subtask['people'],
                "Start_time": subtask['startTime'],
                "End_time": subtask['endTime']
            }
            output["Subtasks"].append(subtask_info)
        
        print("生成行程表")
        print(output)

        # get schedule info (TBD)
        pre_df = pd.DataFrame(output)
        print(pre_df)

        # x-axis variable
        df = pd.json_normalize(pre_df['Subtasks'])[['Start_time', 'End_time', 'Task']]
        print(df)
        proj_start = df.Start_time.min()
        print(proj_start)

        df['start_num'] = (pd.to_datetime(df.Start_time, format='%m-%d') - pd.to_datetime(proj_start, format='%m-%d')).dt.days
        df['end_num'] = (pd.to_datetime(df.End_time, format='%m-%d') - pd.to_datetime(proj_start, format='%m-%d')).dt.days
        df['days_start_to_end'] = df.end_num - df.start_num


        # drawing
        fig, ax = plt.subplots(1, figsize=(16,5))
        ax.barh(df.Task, df.days_start_to_end, left=df.start_num)

        # Ticks
        xticks = np.arange(0, df. end_num.max()+1, 1)
        xticks_labels = pd.date_range(pd.to_datetime(proj_start, format='%m-%d'), end=pd.to_datetime(df.End_time, format='%m-%d').max()).strftime("%m/%d")
        xticks_minor = np.arange(0, df.end_num.max()+1, 1)
        ax.set_xticks(xticks)
        ax.set_xticks(xticks_minor, minor=True)
        ax.set_xticklabels(xticks_labels[::1])

        # plt.show()
        plt.gca().invert_yaxis()
        plt.savefig('task_chart.png')
        client_id = '7ece996e29fd7dc'
        client_secret = '76157f82924e2ab9583e8d018cd6d01a5b767e75'
        client = ImgurClient(client_id, client_secret)

        image_path = 'task_chart.png'
        uploaded_image = client.upload_from_path(image_path, anon=True)

        image_url = uploaded_image['link']
        output["ImgUrl"] = image_url
        print(image_url)
        print(output)

        return output

    args_schema: Optional[Type[BaseModel]] = ScheduleGenerateInput
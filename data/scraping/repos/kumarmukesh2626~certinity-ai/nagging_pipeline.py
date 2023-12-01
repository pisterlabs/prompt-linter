import json
import pandas as pd
import openai
import pandas as pd
from database.ConnectDb import DatabaseHandler
from datetime import datetime
# from utils import get_pdf_file_names_from_db
from src.receiveResponse.fetch_mail import pipeline
# from utils import fetchCompletionData

db = DatabaseHandler()

from datetime import datetime,timedelta

# Calculate the start and end dates and times
current_date = datetime.now()
endDate = current_date.strftime('%Y-%m-%d')
startDate = (current_date - timedelta(days=2)).strftime('%Y-%m-%d')
startTime = "00:00:00"
endTime = "23:59:00"
start_date_filter = datetime.strptime(startDate, '%Y-%m-%d').date()
end_date_filter = datetime.strptime(endDate, '%Y-%m-%d').date()

start_time_filter = datetime.strptime(startTime, '%H:%M:%S').time()
end_time_filter = datetime.strptime(endTime, '%H:%M:%S').time()

try:
    received_data = pipeline(start_date_filter, end_date_filter, start_time_filter, end_time_filter)
    # print(received_data)
except Exception as e:
    print(e)
subject = received_data['subject']

import re

pattern  = r'(?P<projectDescription>.+)-(?P<timesheetEffort>\d+\.\d+)'
# pattern = r'^Re:\s+(?P<projectDescription>.+)-(?P<timesheetEffort>\d+\.\d+)'
print(subject)
# Use re.search to find the matches
match = re.search(pattern, str(subject))
print(match)
# Check if a match was found
if match:
    
    project_description = match.group('projectDescription').strip()
    project_description = re.sub(r'^\d+\s*Re:', '', project_description).strip()
    # project_description = project_description.lstrip('Re: ').strip()
    timesheet_effort = match.group('timesheetEffort')
    try:
        query1 = """ SELECT
            
            spocMailid,
            spocName,
            timeSheetEffort,
            projectDescription,
            taskType
        FROM
            naggingDetails
        WHERE
            timesheetEffort IS NOT NULL
            AND projectDescription IS NOT NULL
            AND taskType = 'Pending'
            AND (projectDescription LIKE '%{project_description}%')
        """.format(project_description=project_description)

        current_timestamp = datetime.now()
        data = db.fetch_data(query1)
        if not data.empty:
            for index, row in data.iterrows():
                spocMailid = row['spocMailid']
                timeSheetEffort = row['timeSheetEffort']
                projectDescription = row['projectDescription']
                taskType = row['taskType']
                receiver_mail = received_data['sender']
                for receiverMail in receiver_mail:
                    if (spocMailid == receiverMail ) and (timeSheetEffort == float(timesheet_effort)): 
                        sbj = project_description + '-' +  str(timesheet_effort)
                        matching_rows = received_data[received_data['subject'].str.contains(sbj, case=False)]
                        if matching_rows['repliedByEmployee'].any():
                            update_dbQuery = """
                                UPDATE naggingDetails
                                SET Response = '{response}'
                                WHERE spocMailid = '{spocMailid}'
                                AND timeSheetEffort = {timeSheetEffort}
                                AND projectDescription = '{projectDescription}'
                            """.format(response=matching_rows['repliedByEmployee'], spocMailid=receiverMail, timeSheetEffort=timesheet_effort,projectDescription=project_description)

                            db.execute_query(update_dbQuery)

        else:
            print("No matching data found.")
        
    except Exception as e:
        print("Error",e)
else:
    print("No match found.")

taskDescription = received_data['repliedByEmployee']

for task in taskDescription:
    feedback =  "RND" #fetchCompletionData(task)
    received_data['gptFeedback'] = feedback
# print(received_data)

    # db.run_df('nagTable',received_data)
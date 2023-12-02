import json
import pandas as pd
import openai
import pandas as pd
from database.ConnectDb import DatabaseHandler
from datetime import datetime
# from utils import get_pdf_file_names_from_db
from fetch_mail import pipeline
from utils import fetchCompletionData

db = DatabaseHandler()

from datetime import datetime

startDate = "2023-08-22"
startTime = "00:00:00"
endDate = "2023-09-11"
endTime = "23:59:00"

start_date_filter = datetime.strptime(startDate, '%Y-%m-%d').date()
end_date_filter = datetime.strptime(endDate, '%Y-%m-%d').date()

start_time_filter = datetime.strptime(startTime, '%H:%M:%S').time()
end_time_filter = datetime.strptime(endTime, '%H:%M:%S').time()

received_data = pipeline(start_date_filter, end_date_filter, start_time_filter, end_time_filter)
taskDescription = received_data['repliedByEmployee']

for task in taskDescription:
    feedback =  "RND" #fetchCompletionData(task)
    received_data['gptFeedback'] = feedback
print(received_data)

    # db.run_df('nagTable',received_data)
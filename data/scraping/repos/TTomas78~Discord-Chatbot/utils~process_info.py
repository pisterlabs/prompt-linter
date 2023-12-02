from langchain.schema import Document

from datetime import datetime
import csv

def process_info(file_name):
  with open(file_name, 'r', encoding="utf8") as file:
    reader = csv.DictReader(file)
    conversations = []
    last_date = None
    row_number = 1
    for row in reader:
      datetime_object = datetime.strptime(row['Date'], '%m/%d/%Y %I:%M %p')
      if last_date is None:
        last_date = datetime_object
        storage = [row]
      else:
        difference = (datetime_object - last_date).total_seconds() // 3600
        if difference > 5:
          #means that have passed 5 hours since the last message, so it's a new conversation
          #have to save the whole conversation
          last_date = datetime_object
          if len(str(storage)) < 3500:
            #needs to convert the message to a string
            storage_str = ["Date:{}\nUser:{}\nMessage:{}".format(info['Date'],info['Author'],info['Content']) for info in storage]
            storage_str_joined = '\n'.join(storage_str)
            document = Document(page_content=storage_str_joined, metadata={'source': '/content/output.csv', 'row': row_number})
            row_number +=1
            conversations.append(document)
          storage = [row]
        else:
          last_date = datetime_object
          storage.append(row)

  return conversations
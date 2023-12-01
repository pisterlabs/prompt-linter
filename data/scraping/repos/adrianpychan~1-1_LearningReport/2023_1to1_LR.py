from googleapiclient import discovery
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from datetime import datetime
import pandas as pd
from datetime import datetime
from templates import templates_ID
import openai

SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]

creds = Credentials.from_authorized_user_file("token.json", SCOPES)
drive_service = build("drive", "v3", credentials=creds)
service = build("docs", "v1", credentials=creds)
gs_service = discovery.build('sheets', 'v4', credentials=creds)

#1iRi26Bs936wv0rVDBLFBHtaFTK9Wg38_ichcNVD4hhQ

LR_sheet = "1iRi26Bs936wv0rVDBLFBHtaFTK9Wg38_ichcNVD4hhQ"
parent_folder = "1j3g4rD-sZ0Xm1reSPDiuDDOP7c4hGzbw"

#SPREADSHEET INFO
body = gs_service.spreadsheets().values().get(spreadsheetId= LR_sheet, majorDimension = "ROWS", range= "B1:K").execute()["values"]

#Creating a DataFrame based on Existing Spreadsheet (LR_sheet) information
df = pd.DataFrame(body[1:], columns = body[0])

def chatGPT(name):
  api_key = "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"
  openai.api_key = api_key
  response = openai.Completion.create(
  engine = "text-davinci-003",   
  prompt = f"In three paragraphs (each paragraph containing 40 words each), praise {name.split()[0]}'s learning ability, learning attitude and nice feedback in class without a chatgpt tone",     
  max_tokens = 512,               
  temperature = 1,               
  top_p = 0.75,                   
  n = 1,)

  completed_text = response["choices"][0]["text"]
  return completed_text.strip()
  
def doc_replace(name, course_name, module):
  requests = [
        {"replaceAllText": 
         {"containsText": 
          {"text": "{{name}}", "matchCase": "true"},
          "replaceText": f"{name}",}
        },
        {
            "replaceAllText": {
                "containsText": {"text": "{{module}}", "matchCase": "true"},
                "replaceText": f"{module}",
            }
         },
        {
            "replaceAllText": {
                "containsText": {"text": "{{date}}", "matchCase": "true"},
                "replaceText": datetime.now().strftime("%B %Y"),
            }
         },
        {
            "replaceAllText": {
                "containsText": {"text": "{{learning_report}}", "matchCase": "true"},
                "replaceText": f"comment here",
            }
         },
         ]
  month = datetime.now().strftime("%B")[0:3]
  year = datetime.now().year
  body = {
            "name": f"{name}_{course_name}_Learning Report_{month} {year}",
            "mimeType": "application/vnd.google-apps.document",
            "parents": [parent_folder],
        }
  
  doc = (drive_service.files().copy(fileId = templates_ID[course_name], body=body).execute())
  doc_id = doc.get("id")
  print(f'\nCreated document with title: {body["name"]}. \nURL: https://docs.google.com/document/d/{doc_id}/\n')
  format_result = (
            service.documents().batchUpdate(documentId = doc_id ,body={"requests": requests}).execute()
        )
  return f"https://docs.google.com/document/d/{doc_id}/edit"

def update_LRsheet(cell_range, column_title, values):

  #Clear Sheet to replace
  gs_service.spreadsheets().values().clear(
    spreadsheetId= LR_sheet, 
    range = cell_range,
    body = {},).execute()

  #Replacing Existing Gsheet with Updates Values:
  gs_service.spreadsheets().values().update(
    spreadsheetId= LR_sheet, 
    range = cell_range,
    valueInputOption = "USER_ENTERED",
    body = {"values" : [column_title] + values},).execute()

for status in range(len(df["Generation Status"])):
  if df["Generation Status"][status] == "Incomplete" and df["Course Name"][status] in templates_ID.keys():  
    df["Learning Report Link"][status] = doc_replace(
      df["Student Name"][status],
      df["Course Name"][status],
      df["Module"][status])

    df["Task Date Completed"][status] = datetime.now().strftime("%Y-%m-%d")

  df["Generation Status"][status] = f'=IF(ISBLANK(H{status+2}), "Incomplete", "Complete")'

update_LRsheet("B1:K", df.columns.values.tolist(), df.values.tolist())
import json
from dataclasses import dataclass
import os
import pandas as pd
import gdown
from openai import OpenAI
from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table
from sqlalchemy.orm import sessionmaker

# Load environment variables
from dotenv import load_dotenv


descr = "Ищет соотвтествующий вопрос если не нашел соотвтествия - возвращает пустоту"

load_dotenv()

# Set up database connection
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)

metadata = MetaData()
avatarexsettings = Table(
    'home_avatarexsettings', metadata,
    Column('id', Integer, primary_key=True),
    Column('knowledge_link', String),
    Column('context', String),
    Column('api_token', String),
    Column('error_message', String),
)

# Создаем сессию
Session = sessionmaker(bind=engine)
session = Session()



@dataclass()
class AvatarexSettings:
    knowledge_link: str
    context: str
    api_token: str
    error_message: str


def get_execution_function(filename):
    df = pd.read_excel(filename)
    first_row = list(df.iloc[:, 0])
    properties = {}
    rq = []
    for r in first_row:
        if r in rq:
            continue
        rq.append(r)
        properties[r] = {'type': 'boolean', 'description': 'Вопрос полностью соответствует заданному?'}

    return [{
        "name": "get_question_by_context",
        "description": descr,
        "parameters": {
            "type": "object",
            "properties": properties,
            'required': rq
        }
    }]


def read_avatarex_settings() -> AvatarexSettings:
    # Read data from the table where id = 2
    result = session.query(avatarexsettings).filter(avatarexsettings.c.id == 2).first()
    return AvatarexSettings(
        knowledge_link=result.knowledge_link,
        context=result.context,
        api_token=result.api_token,
        error_message=result.error_message
    )

def download_file(db_name):
    file_id = db_name.replace('https://docs.google.com/spreadsheets/d/', '')
    file_id = file_id.split('/')[0]
    try:
        os.remove(f"uploads/{file_id}.xlsx")
    except OSError:
        pass
    try:
        download_url = f"https://drive.google.com/uc?id={file_id}"
        output_path = f"uploads/{file_id}.xlsx"
        gdown.download(download_url, output_path, quiet=True)
    except:
        pass
    return output_path


def get_keywords_values(message, func):
    client = OpenAI()
    try:
        messages = [
            {'role': 'system', 'content': descr},
            {"role": "user",
             "content": message}]

        response = client.chat.completions.create(model="gpt-3.5-turbo",
                                                  messages=messages,
                                                  functions=func,
                                                  function_call={"name": "get_question_by_context"})
        response_message = response.choices[0].message
    except Exception as e:
        print("ERROR", e)
        return {'is_ok': False, 'args': {}}
    if response_message.function_call:
        function_args = json.loads(response_message.function_call.arguments)
        try:
            return {'is_ok': True, 'args': list(function_args.keys())}
        except:
            return {'is_ok': False, 'args': []}
    else:
        return {'is_ok': False, 'args': []}


def get_answer_by_question(questions, filename):
    answer = ''
    try:
        df = pd.read_excel(filename)
        list_of_arrays = list(df.iloc)

        for i in list_of_arrays:
            if questions.strip().lower() in i[0].strip().lower():
                answer += str(i[1]) + '\n'
                break
    except Exception as e:
        print(e)
    return answer
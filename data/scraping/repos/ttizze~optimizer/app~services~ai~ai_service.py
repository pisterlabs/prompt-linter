from dotenv import load_dotenv

load_dotenv()
import os
from pydantic import Field, BaseModel

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
import guidance


class AIService(BaseModel):
    def profile_from_personal_information(self, query: str) -> str:
        llm = guidance.llms.OpenAI("gpt-3.5-turbo")
        profile_from_personal_information = guidance("""
            {{#system~}}
            あなたはソーシャルワーカーです。ある個人情報を読み、現在置かれている状況を推察し、支援が受けられそうな特徴をリスト形式で出力してください。
            {{~/system}}

            {{#user~}}
            下記が私の個人情報です。
            {{query}}
            {{~/user}}

            {{#assistant~}}
            {{gen 'support_feature' temperature=0 max_tokens=500}}
            {{~/assistant}}

            """)
        response =profile_from_personal_information(query=query,llm=llm)

        return response['support_feature']


    def update_profile(self,profile:str,message:str) -> str:
        llm = guidance.llms.OpenAI("gpt-3.5-turbo")
        update_profile = guidance("""
            {{#system~}}
            あなたはソーシャルワーカーです。ユーザーの個人情報と、ユーザーの現在困っていることを読み、現在置かれている状況を推察し、支援が受けられそうな特徴をリスト形式で出力してください。
            {{~/system}}
            {{#user~}}
            下記に個人情報を貼ります。
            {{profile}}

            下記が現在困っていることです。
            {{message}}

            上記の個人情報に、現在私が困っていることを要約し、書き加えて出力してください。
            {{~/user}}

            {{#assistant~}}
            {{gen 'support_feature' temperature=0 max_tokens=500}}
            {{~/assistant}}

            """)
        response =update_profile(profile=profile,message=message,llm=llm)
        print(response)

        return response['support_feature']
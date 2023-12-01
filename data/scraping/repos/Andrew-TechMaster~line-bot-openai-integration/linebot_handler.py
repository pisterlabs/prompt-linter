import re
import os
import openai
from linebot.models import TextSendMessage, TemplateSendMessage, ConfirmTemplate, PostbackAction
from dotenv import load_dotenv

# Set up OpenAI API key
load_dotenv()  # Load environment variables from .env file
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# <==================== Message Handler ====================> #
class LinebotHandler:
    def __init__(self):
        self.options_dict = {
            '味ポン': ['なすのさっぱり焼きびたし', '旨ダレやみつき大葉ナス', 'きゃべつのナルム風サラダ'],
            'ごまぽん': ['夏野菜のしゃぶしゃぶ', '豆腐サラダ'],
        }
        self.workflow_status = {}

    def handle_postback(self, user_id, data, event, line_bot_api):
        if user_id in self.workflow_status:
            if data == 'はい':
                response = self.chat_gpt(
                    "workflow", self.workflow_status[user_id], role="user")
                self.reply_with_text(line_bot_api, event.reply_token, response)
                self.workflow_status.pop(user_id, None)
            elif data == 'いいえ':
                self.reply_with_text(
                    line_bot_api, event.reply_token, "申し訳ございませんが、その質問にはお答えすることができません。")
                self.workflow_status.pop(user_id, None)

    def handle_message(self, event, line_bot_api):
        user_id = event.source.user_id
        message = event.message.text

        if user_id in self.workflow_status:
            self.workflow_status[user_id] = message
            self.handle_workflow_response(user_id, line_bot_api, event)
        else:
            if re.match("gpt4", message):
                self.workflow_status[user_id] = "ask_mood"
                self.reply_with_text(
                    line_bot_api, event.reply_token, "Hello, I am chatgpt4! 今日の気分を教えてください！")
            else:
                self.handle_basic_cases(message, event, line_bot_api)

    def handle_workflow_response(self, user_id, line_bot_api, event):
        if self.workflow_status[user_id]:
            self.reply_with_confirm_template(
                line_bot_api, event.reply_token, "味ぽん?")

    def handle_basic_cases(self, message, event, line_bot_api):
        for keyword, options in self.options_dict.items():
            if re.match(keyword, message):
                result = self.chat_gpt("basic_case", keyword)
                self.reply_with_text(line_bot_api, event.reply_token, result)
                return
        self.reply_with_text(line_bot_api, event.reply_token, 'おすすめのレシピはありません')

    def chat_gpt(self, context, mood_or_keyword, role="user", max_words=400):
        message_history = []

        if context == "workflow":
            user_input = f"味ぽんを使った{mood_or_keyword}の人に最適なレシピを1つだけ簡潔に教えてください"
        elif context == "basic_case":
            user_input = f"ミツカンの{mood_or_keyword}をつかったレシピを1つだけ簡潔に教えてください"

        message_history.append({"role": role, "content": user_input})

        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=message_history,
        )

        reply_content = completion.choices[0].message.content
        return reply_content

    def reply_with_text(self, line_bot_api, reply_token, message):
        line_bot_api.reply_message(reply_token, TextSendMessage(text=message))

    def reply_with_confirm_template(self, line_bot_api, reply_token, text):
        line_bot_api.reply_message(
            reply_token,
            TemplateSendMessage(
                alt_text="はいかいいえを選んでください",
                template=ConfirmTemplate(
                    text=text,
                    actions=[
                        PostbackAction(
                            label="はい",
                            data="はい",
                        ),
                        PostbackAction(
                            label="いいえ",
                            data="いいえ",
                        ),
                    ],
                ),
            ),
        )

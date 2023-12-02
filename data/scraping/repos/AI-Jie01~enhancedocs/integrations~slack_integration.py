import main
import utils

from slack_sdk import WebClient
from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler
from slack_sdk.errors import SlackApiError
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.chains import ConversationalRetrievalChain


class SlackClient:

    def mention_handler(self, body, say):
        text = body["event"].get("text")
        channel = body["event"].get("channel")
        thread_ts = body["event"].get("thread_ts")
        ts = body["event"].get("ts")
        if text:
            if utils.is_db_empty(main.config):
                say(
                    text="No data found. Contact your Slack administrator. "
                         "If you are the administrator; ingest data using "
                         "https://github.com/enhancedocs/cli or the API directly",
                    channel=channel,
                    thread_ts=thread_ts
                )
                return
            say(
                text="Hey there! Give me a moment... "
                     "Let me look trough the documentation and I'll come back with an answer!",
                channel=channel,
                thread_ts=ts
            )
            store = utils.get_vector_store(main.config)
            prompt = main.config.prompt
            question = text.split(f'<@{self.user_id}> ', 1)[1]
            if thread_ts:
                replies = self.slack_web.conversations_replies(channel=channel, ts=thread_ts, limit=100)
                messages = replies.get("messages", [])
                chat_history = []

                for message in messages:
                    if message.get("bot_id"):
                        chat_history.append(f"AI: {message['text'].replace(f'<@{self.user_id}>', 'AI')}")
                    else:
                        chat_history.append(f"User: {message['text'].replace(f'<@{self.user_id}>', 'AI')}")

                print('\n'.join(chat_history))
                question_generator = LLMChain(llm=main.llm, prompt=CONDENSE_QUESTION_PROMPT)
                doc_chain = load_qa_with_sources_chain(main.llm, chain_type="stuff", prompt=prompt)
                chain = ConversationalRetrievalChain(
                    combine_docs_chain=doc_chain,
                    retriever=store.as_retriever(),
                    question_generator=question_generator,
                    get_chat_history=utils.get_chat_history,
                    return_source_documents=True
                )
                result = chain(
                    {"question": question, "project_name": main.config.project_name, "chat_history": chat_history},
                    return_only_outputs=True
                )
                say(
                    text=result.get("answer"),
                    channel=channel,
                    thread_ts=thread_ts
                )
            else:
                doc_chain = load_qa_with_sources_chain(main.llm, chain_type="stuff", prompt=prompt)
                chain = RetrievalQAWithSourcesChain(
                    combine_documents_chain=doc_chain,
                    retriever=store.as_retriever(),
                    return_source_documents=True
                )
                result = chain(
                    {"question": question, "project_name": main.config.project_name},
                    return_only_outputs=True
                )
                say(
                    text=result.get("answer"),
                    channel=channel,
                    thread_ts=ts
                )

    def __init__(self, slack_bot_token, slack_signing_secret):
        self.slack_web = WebClient(token=slack_bot_token)
        self.slack_app = App(token=slack_bot_token, signing_secret=slack_signing_secret)
        self.handler = SlackRequestHandler(self.slack_app)
        try:
            auth_result = self.slack_web.auth_test()
            self.user = auth_result.get("user")
            self.user_id = auth_result.get("user_id")
            self.slack_app.event("app_mention")(self.mention_handler)
            print(f'[Slack]: Logged in as {self.user} (ID: {self.user_id})')
        except SlackApiError:
            print("Failed to authenticate Slack client")

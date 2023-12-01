import os
import sys
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from persistence import Persistence
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import logging
import logging.handlers

logger = logging.getLogger("")
logger.setLevel(logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
If there are two or three likely answers, list all of the likely answers.
Use three sentences maximum and keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Initializes your app with your bot token and socket mode handler
app = App(token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET"))

llm = ChatOpenAI(model_name="gpt-3.5-turbo", 
                 temperature=0)

vectordb = Persistence.get_storage('index_these')

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    #verbose=True,
    memory=ConversationBufferWindowMemory(k=2),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

app.event("app_home_opened")
def update_home_tab(client, event, logger):
  try:
    # views.publish is the method that your app uses to push a view to the Home tab
    client.views_publish(
      # the user that opened your app's app home
      user_id=event["user"],
      # the view object that appears in the app home
      view={
        "type": "home",
        "callback_id": "home_view",

        # body of the view
        "blocks": [
          {
            "type": "section",
            "text": {
              "type": "mrkdwn",
              "text": "*Welcome to your _App's Home_* :tada:"
            }
          },
          {
            "type": "divider"
          },
          {
            "type": "section",
            "text": {
              "type": "mrkdwn",
              "text": "This button won't do much for now but you can set up a listener for it using the `actions()` method and passing its unique `action_id`. See an example in the `examples` folder within your Bolt app."
            }
          },
          {
            "type": "actions",
            "elements": [
              {
                "type": "button",
                "text": {
                  "type": "plain_text",
                  "text": "Click me!"
                }
              }
            ]
          }
        ]
      }
    )

  except Exception as e:
    logger.error(f"Error publishing home tab: {e}")

# @app.event("message")
# def handle_message_events(body, logger):
#     logger.info(body)

#Message handler for Slack
#@app.message(".*")
@app.event("message")
def message_handler(message, say, logger):
    try:
        if "text" in message:
            logger.debug("Chat: " + message['text'])
            output = qa_chain({"query": message['text']})
            logger.debug(output)
            say(output['result'])
        else:
            logger.debug('Unknown message type')
    except Exception as e:
        logger.error(f"Error responding to chat message: {e}")
        #say(f"Error responding to chat message: {e}")

# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"], 
                    #   logger=logger,
                    #   trace_enabled=True,
                    #   all_message_trace_enabled = False,
                    #   ping_pong_trace_enabled = False
                      ).start()
    #app.start(port=int(os.environ.get("PORT", 3000)))

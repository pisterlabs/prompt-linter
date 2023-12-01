from langchain.memory import PostgresChatMessageHistory

history = PostgresChatMessageHistory(
    connection_string="postgresql://postgres:mypassword@localhost/chat_history",
    session_id="foo",
)

history.add_user_message("hi!")

history.add_ai_message("whats up?")
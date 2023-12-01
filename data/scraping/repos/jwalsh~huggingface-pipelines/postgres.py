from langchain.memory import PostgresChatMessageHistory
import os

POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
history = PostgresChatMessageHistory(
    connection_string=f"postgresql://postgres:{POSTGRES_PASSWORD}@localhost:5432/chat_history",
    table_name="message_store",
    session_id="foo",
)

history.add_user_message("hi!")

history.add_ai_message("whats up?")

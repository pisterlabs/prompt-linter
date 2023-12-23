from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

INPUT_VARIABLES = ["input", "chat_history", "agent_scratchpad", "bot_id", "channel_id", "time_message", "context"]

SYSTEM_MESSAGE = """\
Assistant is a Slack bot with ID {bot_id}, operating in channel {channel_id}, responding within a specific thread.

Mention users as <@USER_ID> and link channels as <#CHANNEL_ID> in Slack mrkdwn format. {time_message}

Always include permalinks in the final answer when available and adhere to user-defined context.

USER-DEFINED CONTEXT
====================
{context}

CONVERSATIONS IN THE CURRENT THREADS
====================================\
"""

HUMAN_MESSAGE = """\
TOOLS
-----
Assistant can provide an answer based on the given inputs. \
However, if needed, the human can use tools to look up additional information \
that may be helpful in answering the user's original question. The tools the human can use are:

{{tools}}

{format_instructions}

LAST USER'S INPUT
-----------------
Here is the user's last input \
(remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):

{{{{input}}}}\
"""

SLACK_CONVERSATION_SEARCH_NAME = "Slack Conversation Search"

SLACK_CONVERSATION_SEARCH_DESCRIPTION = """\
A tool for referencing information from past conversations outside the current thread. \
Useful for when an answer may be in previous discussions, attached files, or unfurling links. \
Avoid mentioning that you used this tool in the final answer. \
Present the information as if it were organically sourced instead. \
Input should be a question in natural language that this tool can answer.\
"""

SLACK_URL_SEARCH_NAME = "Slack URL Search"

SLACK_URL_SEARCH_DESCRIPTION = """\
A tool for extracting precise information from URLs that have been shared within Slack conversations. \
This includes unfurling links, attached files, or even other messages that have been referenced in Slack messages. \
Useful for when you need to retrieve detailed data from a specific URL previously mentioned in a conversation. \
Input should be a URL (i.e. https://www.example.com).\
"""

QUESTION_SYSTEM_MESSAGE = """\
Use the following portion of a long document to see if any of the text is relevant to answer the question.
Return any relevant text verbatim.
When providing your answer, consider the timestamp, channel, user, \
and page which may not align with the original document.
Always include the permalink in your response.
----------------
{context}\
"""

QUESTION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(QUESTION_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

COMBINE_SYSTEM_MESSAGE = """\
Given the following extracted parts of a long document and a question, create a final answer.
Consider the timestamp, channel and user when providing your answer.
Always include the permalink in your response.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
______________________
{summaries}\
"""

COMBINE_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(COMBINE_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

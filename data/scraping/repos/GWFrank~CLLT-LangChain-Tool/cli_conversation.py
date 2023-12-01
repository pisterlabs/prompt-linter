import readline
import atexit

from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

import tools

VERBOSE = False


class Colors:
    # ANSI Color Code Reference: https://gist.github.com/rene-d/9e584a7dd2935d0f461904b9f2950007
    """ ANSI color codes """
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"


def load_agent():
    print("Loading agent...")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                     streaming=True,
                     verbose=VERBOSE,
                     temperature=0,
                     client=None,)
    memory = ConversationBufferMemory(memory_key="chat_history",
                                      return_messages=True)
    tool_list = [tools.GetPttPostsKeywordsOnDate(),
                 tools.GetKeywordsVote(),
                 tools.GetKeywordsVoteTrend(),
                 tools.GetUpvoteCommentsByKeyword(),
                 tools.GetDownvoteCommentsByKeyword(),

                 tools.GetPostIDsByDate(),
                 tools.GetPostKeywordsByID(),
                 tools.GetPostTitleByID(),
                 tools.GetUpvoteCountByID(),
                 tools.GetDownvoteCountByID(),
                 tools.GetArrowCountByID(),

                 tools.GetNewsTitlesWithCrawler(),
                 tools.GetNewsKeywordsWithCrawler(),
                 ]
    agent = initialize_agent(tool_list,
                             llm,
                             #   agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                             memory=memory,
                             handle_parsing_errors=True,
                             verbose=VERBOSE)

    return agent


def talk_to_agent(agent, msg: str):
    print("Thinking...")
    ret = agent.run(msg)
    print(f"{Colors.LIGHT_BLUE}LLM: {ret}{Colors.END}")


def main():
    print("==== CLLT Final Project - Demo ====")
    load_dotenv('.env')
    agent = load_agent()

    system_msg = """System: 你是一個臺灣 PTT 使用者及政治觀察家，請使用提供的 tools 完成後面提供給你的工作，並使用臺灣的中文回答問題。
有些提供的 tool 完全不會使用到，但是你可以自己決定要不要使用。
請先和使用者打個招呼吧！"""
    print(f"{Colors.YELLOW}{system_msg}{Colors.END}")
    talk_to_agent(agent, system_msg)

    while True:
        msg = input(f"{Colors.LIGHT_GREEN}>>> You: ")
        print(Colors.END, end="")
        talk_to_agent(agent, msg)


if __name__ == "__main__":
    atexit.register(lambda: print(f"{Colors.LIGHT_RED}Bye!{Colors.END}"))
    main()

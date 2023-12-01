import re
import random
from langchain.prompts import PromptTemplate
from modules.gpt_modules import gpt_call

from .normal_debate import nomal_debator
from .one_to_one_debate import one_to_one_debator


#############################################
# Debate bot setting
#############################################
def debate_bot(prompt, history="", debate_subject="", bot_role="", history_num=0):

    if bot_role == "토론":
        #bot_response = nomal_debator(prompt, history, debate_subject, bot_role, history_num)
        bot_response = one_to_one_debator(prompt, history, debate_subject, bot_role, history_num)
    elif bot_role == "주제 정의":
        pass
    elif bot_role == "POI 연습":
        pass
    elif bot_role == "역할 추천":
        pass
    elif bot_role == "주장 비판":
        pass
    else:
        print("bot_role error")
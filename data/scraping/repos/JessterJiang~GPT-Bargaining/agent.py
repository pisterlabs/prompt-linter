# -------------------------------------------------------------------------------------------------------------------- #
# 导入

# 导入四个大语言模型
import openai  # 导入openai模块，用于与GPT模型进行交互和访问
import anthropic  # 导入anthropic模块，可能是用于与Anthropic API进行交互的库
import ai21  # 导入ai21模块，可能是用于与ai21 API进行交互的库
import cohere  # 导入cohere模块，可能是用于与Cohere API进行交互的库

# 导入各种处理库
import re  # 导入re模块，这是Python的正则表达式模块，用于处理文本的匹配和提取
from copy import deepcopy  # 导入深拷贝函数deepcopy，用于复制数据
from pprint import pprint  # 导入漂亮打印函数pprint，用于以更美观的方式打印数据
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed

# 导入自定义库
from lib_api import *  # 导入自定义模块lib_api中的所有内容


# from local.azure import azure_completion_with_backoff


# -------------------------------------------------------------------------------------------------------------------- #
# 将包含角色和对话内容的文本文件，解析成对应的Python字典列表
# 输入：聊天记录的文件名（字符串）
def load_initial_instructions(path_to_instructions):
    """Load initial instructions from textual format to a python dict"""

    pattern = r"==== (SYSTEM|USER|ASSISTANT)"  # 定义一个正则表达式模式，用于匹配不同角色的标记

    # 使用正则表达式将文本内容拆分成角色和对话内容的列表
    with open(path_to_instructions) as f:
        content = f.read()  # 读取整个文本内容
        content = re.split(pattern, content)  # 使用正则表达式模式拆分文本内容
        content_ = []
        for c in content:
            if c != "": content_.append(c)  # 去除空白内容并添加到新的列表中
        content = content_  # 更新文本内容为处理后的列表内容
        l = len(content)  # 获取拆分后的内容列表长度
        assert (l % 2 == 0)  # 确保内容列表的长度是偶数（角色和对话内容应该成对出现）
        initial_instruction = []
        for i in range(0, l, 2):
            instruction = {
                "role": content[i].strip().lower().replace("====", "").replace(" ", "").strip(),
                # 获取角色，去除首尾空格并转换为小写，并移除多余的"===="和空格
                "content": content[i + 1].strip()  # 获取对话内容，去除首尾空格
            }
            initial_instruction.append(instruction)  # 将角色和对话内容组成字典，并添加到初始指令列表中
    return initial_instruction  # 返回初始指令的Python字典列表


# -------------------------------------------------------------------------------------------------------------------- #
# 这个函数用于判断是否需要调解者介入
# 输入两个角色的聊天(字符串)
def involve_moderator(player_1_run, player_2_run):
    """If at least one player's response does not contain a number, involve a moderator
    The moderator determines if they players have reached an agreement, or break the
    negotiation, or is still in negotiation.
    """

    # 正则表达式，表示整数和实数
    number_pattern = r"[-+]?\d*\.\d+|\d+"

    # 使用正则表达式匹配数字的模式 得到两人说的数字
    match_1 = re.search(number_pattern, player_1_run)
    # print(match_1)
    match_2 = re.search(number_pattern, player_2_run)
    # print(match_2)

    # 判断是否至少有一方的回应不包含数字
    if (match_1 is not None and match_2 is None) or (match_1 is None and match_2 is not None) or (
            match_1 is None and match_2 is None):
        return True  # 两方没有同时说价格，返回True
    else:
        return False  # 双方都在说价格，返回False


# -------------------------------------------------------------------------------------------------------------------- #
# 从聊天记录中解析出历史价格
def parse_final_price(dialog_history):
    """从对话历史中解析出最终价格"""
    # 定义匹配货币价格的正则表达式模式
    money_pattern = r'\$\d+(?:\.\d+)?'

    # 从对话历史的最后一条消息开始逆向遍历
    for d in dialog_history[::-1]:
        # 使用 re.findall() 查找匹配模式的所有结果
        match = re.findall(money_pattern, d["content"])
        if len(match) >= 1:
            # 如果找到匹配结果，则表示可能是价格，取最后一个匹配结果作为最终价格
            final_price = match[-1]

            # 判断最终价格的格式，如果以美元符号 "$" 开头，则去掉美元符号并转换为浮点数
            if final_price[0] == "$":
                final_price = float(final_price[1:])
            else:
                final_price = float(final_price)
            return final_price

    # 如果未找到最终价格，返回 -1 表示解析失败
    return -1


# -------------------------------------------------------------------------------------------------------------------- #
# 对话代理类
class DialogAgent(object):
    """GPT Agent base class, later derived to be a seller, buyer, critic, or moderator

    TODO: add code to detect price inconsistency to seller and buyer
    TODO: release the restriction of the fixed initial price
    """

    def __init__(self,
                 initial_dialog_history=None,
                 agent_type="",  # "seller", "buyer", "critic", "moderator"
                 system_instruction="You are a helpful AI assistant",
                 engine="gpt-3.5-turbo",
                 api_key="",
                 item="balloon"
                 ):
        """Initialize the agent"""
        super().__init__()

        self.agent_type = agent_type
        self.engine = engine
        self.api_key = api_key
        self.item = item

        if ("claude" in self.engine):
            self.claude = anthropic.Client(self.api_key)
        if ("cohere" in self.engine):
            assert self.engine in ["cohere-command-nightly",
                                   "cohere-command",
                                   "cohere-command-light",
                                   "cohere-command-light-nightly"
                                   ]
            self.cohere_model = self.engine[7:]
            self.co = cohere.Client(api_key)

        if (initial_dialog_history is None):
            self.dialog_history = [{"role": "system", "content": system_instruction}]
        else:
            self.initial_dialog_history = deepcopy(initial_dialog_history)
            self.dialog_history = deepcopy(initial_dialog_history)

        self.last_prompt = ""
        return

    def reset(self):
        """Reset dialog history"""
        self.dialog_history = deepcopy(self.initial_dialog_history)
        return

    def call_engine(self, messages):
        """Route the call to different engines"""
        # if("azure" in self.engine):
        #     response = azure_completion_with_backoff(messages=messages)
        #     message = response['choices'][0]['message']
        if ("gpt" in self.engine):
            # import ipdb; ipdb.set_trace()
            response = completion_with_backoff(
                model=self.engine,
                messages=messages
            )
            message = response['choices'][0]['message']
            assert (message['role'] == 'assistant')
        elif ("claude" in self.engine):
            prompt_claude = convert_openai_to_anthropic_prompt(messages)
            # import ipdb; ipdb.set_trace()
            response = claude_completion_with_backoff(self.claude,
                                                      prompt=prompt_claude,
                                                      stop_sequences=[anthropic.HUMAN_PROMPT],
                                                      model=self.engine,
                                                      max_tokens_to_sample=512,
                                                      )
            message = {"role": "assistant", "content": response["completion"].strip()}
        elif ("j2" in self.engine):
            prompt_ai21 = convert_openai_to_ai21_prompt_format_1(messages, self.agent_type)
            # import ipdb; ipdb.set_trace()
            response = ai21_completion_with_backoff(model=self.engine,
                                                    prompt=prompt_ai21,
                                                    numResults=1,
                                                    maxTokens=512,
                                                    temperature=0.7,
                                                    topKReturn=0,
                                                    topP=1,
                                                    stopSequences=["##"]
                                                    )
            content = response["completions"][0]["data"]["text"]
            if (self.agent_type in ["seller", "buyer"]):
                content = content.split('\n')[0]
            message = {"role": "assistant",
                       "content": content
                       }
        elif ("cohere" in self.engine):
            prompt_cohere = convert_openai_to_cohere_prompt(messages)
            # import ipdb; ipdb.set_trace()
            response = cohere_completion_with_backoff(self.co,
                                                      prompt=prompt_cohere,
                                                      model=self.cohere_model,
                                                      max_tokens=512,
                                                      )

            # import ipdb; ipdb.set_trace()
            message = {"role": "assistant",
                       "content": response[0].text
                       }
        else:
            raise ValueError("Unknown engine %s" % self.engine)
        return message

    def call(self, prompt):
        """Call the agent with a prompt. Handle different backend engines in this function
        """
        # TODO: refactor the code, add `remember_history` flag
        #       if yes, then add the prompt to the dialog history, else not
        # 待办：添加remember_history flag，用来判断是否将提示词加入对话历史
        # 将prompt用字典存储，并导入进对话历史
        prompt = {"role": "user", "content": prompt}
        self.dialog_history.append(prompt)
        self.last_prompt = prompt['content']
        # 用的是整个对话历史
        messages = list(self.dialog_history)
        # messages.append(prompt)

        message = self.call_engine(messages)
        # 将call_engine返回的消息写入对话历史
        self.dialog_history.append(dict(message))

        # self.dialog_round += 1
        # self.history_len = response['usage']['total_tokens']
        return message['content']

    @property
    def last_response(self):
        return self.dialog_history[-1]['content']

    @property
    def history(self):
        for h in self.dialog_history:
            print('%s:  %s' % (h["role"], h["content"]))
        return


class BuyerAgent(DialogAgent):

    def __init__(self,
                 initial_dialog_history=None,
                 agent_type="buyer",
                 engine="gpt-3.5-turbo",
                 api_key="",
                 buyer_instruction="buyer",
                 buyer_init_price=10,
                 seller_init_price=20,
                 item="balloon",
                 ):
        """Initialize the buyer agent"""
        super().__init__(initial_dialog_history=initial_dialog_history,
                         agent_type=agent_type,
                         engine=engine,
                         api_key=api_key,
                         item=item,
                         )
        self.buyer_instruction = buyer_instruction
        self.buyer_init_price = buyer_init_price
        self.seller_init_price = seller_init_price

        print("Initializing buyer with engine %s" % self.engine)

        for i, d in enumerate(self.dialog_history):
            self.dialog_history[i]["content"] = d["content"].replace(
                "BUYER_INIT_PRICE", str(buyer_init_price))
            self.dialog_history[i]["content"] = d["content"].replace(
                "SELLER_INIT_PRICE", str(seller_init_price))
        return

    def reset(self):
        """Reset dialog history"""
        self.dialog_history = deepcopy(self.initial_dialog_history)

        for i, d in enumerate(self.dialog_history):
            self.dialog_history[i]["content"] = d["content"].replace(
                "BUYER_INIT_PRICE", str(self.buyer_init_price))
            self.dialog_history[i]["content"] = d["content"].replace(
                "SELLER_INIT_PRICE", str(self.seller_init_price))
        return

    def receive_feedback(self, feedback, previous_price):
        """Receive and acknowledge feedback from the critic
        Basically add the feedback message to the history and restart the bargaining
        """

        # if the previous round is ended by the buyer, then add seller's acknowledgement
        if (self.dialog_history[-1]["role"] == "user"):
            self.dialog_history.append({"role": "assitent", "content": "Sure, happy to do business with you."})

        # add the feedback from the critic
        feedback_prefix = "Well done in your last round. "
        feedback_prefix += "Here is the feedback from the critic:\n\n"
        feedback = feedback_prefix + feedback + "\n\n"
        feedback += "Now let's start the next round. "
        feedback += "In this round, your should try to improve your negotiation strategy based on the feedback from the critic. "
        feedback += "But you are **not allowed** to ask for additionl service. "
        feedback += "Your goal is to buy the %s at at lower price than the previous round, i.e., lower than $%s." % \
                    (self.item, str(previous_price))
        prompt = {"role": "user", "content": feedback}
        self.dialog_history.append(prompt)

        # add the seller's acknowledgement
        acknowledgement = "Sure, I will try to improve my negotiation strategy based on the feedback from the critic."
        acknowledgement += " And I will try to buy it at a lower price (lower than $%s) than the previous round." \
                           % str(previous_price)
        prompt = {"role": "assistant", "content": acknowledgement}
        self.dialog_history.append(prompt)

        # restart the bargaining
        prompt = {"role": "user", "content": "Now ask your price again."}
        self.dialog_history.append(prompt)
        prompt = {"role": "assistant", "content": "Hi, how much is the %s?" % self.item}
        self.dialog_history.append(prompt)
        prompt = {"role": "user",
                  "content": "Hi, this is a good %s and its price is $%d" % (self.item, self.seller_init_price)}
        self.dialog_history.append(prompt)
        if (self.buyer_instruction == "buyer"):
            prompt = {"role": "assistant", "content": "Would you consider selling it for $%d?" % self.buyer_init_price}
            self.dialog_history.append(prompt)
        return acknowledgement


class SellerAgent(DialogAgent):

    def __init__(self,
                 initial_dialog_history=None,
                 agent_type="seller",
                 engine="gpt-3.5-turbo",
                 api_key="",
                 cost_price=10,
                 buyer_init_price=10,
                 seller_init_price=20,
                 item="balloon"
                 ):
        """Initialize the seller agent"""
        super().__init__(initial_dialog_history=initial_dialog_history,
                         agent_type=agent_type,
                         engine=engine,
                         api_key=api_key,
                         item=item,
                         )
        self.seller_init_price = seller_init_price
        self.buyer_init_price = buyer_init_price
        self.cost_price = cost_price

        print("Initializing seller with engine %s" % self.engine)

        for i, d in enumerate(self.dialog_history):
            self.dialog_history[i]["content"] = d["content"].replace("BUYER_INIT_PRICE", str(buyer_init_price))
            self.dialog_history[i]["content"] = d["content"].replace("SELLER_INIT_PRICE", str(seller_init_price))
            self.dialog_history[i]["content"] = d["content"].replace("COST_PRICE", str(cost_price))
        return

    def reset(self):
        """Reset dialog history"""
        self.dialog_history = deepcopy(self.initial_dialog_history)

        for i, d in enumerate(self.dialog_history):
            self.dialog_history[i]["content"] = d["content"].replace("BUYER_INIT_PRICE", str(self.buyer_init_price))
            self.dialog_history[i]["content"] = d["content"].replace("SELLER_INIT_PRICE", str(self.seller_init_price))
            self.dialog_history[i]["content"] = d["content"].replace("COST_PRICE", str(self.cost_price))
        return

    def receive_feedback(self, feedback, previous_price):
        """Receive and acknowledge feedback from the critic
        Basically add the feedback message to the history and restart the bargaining
        """

        # if the previous round is ended by the buyer, then add seller's acknowledgement
        if (self.dialog_history[-1]["role"] == "user"):
            self.dialog_history.append({"role": "assitent", "content": "Sure, happy to do business with you."})

        # add the feedback from the critic
        feedback_prefix = "Well done in your last round. "
        feedback_prefix += "Here is the feedback from the critic:\n\n"
        feedback = feedback_prefix + feedback + "\n\n"
        feedback += "Now let's start the next round. "
        feedback += "In this round, your should try to improve your negotiation strategy based on the feedback from the critic. "
        feedback += "Your goal is to sell the %s at at higher price than the previous round, i.e., higher than $%s." % \
                    (self.item, str(previous_price))
        prompt = {"role": "user", "content": feedback}
        self.dialog_history.append(prompt)

        # add the seller's acknowledgement
        acknowledgement = "Sure, I will try to improve my negotiation strategy based on the feedback from the critic."
        acknowledgement += " And I will try to sell it at a higher price (higher than $%s) than the previous round." % str(
            previous_price)
        prompt = {"role": "assistant", "content": acknowledgement}
        self.dialog_history.append(prompt)

        # restart the bargaining
        prompt = {"role": "user", "content": "Hi, how much is the %s?" % self.item}
        self.dialog_history.append(prompt)
        prompt = {"role": "assistant",
                  "content": "Hi, this is a good %s and its price is $%d" % (self.item, self.seller_init_price)}
        self.dialog_history.append(prompt)
        return acknowledgement


class ModeratorAgent(DialogAgent):
    """NOTE: initial experiments shows that the moderator is much better at recognizing deal than not deal
    Do not know why but interesting
    相对于不成交更容易识别成交，不知道为什么
    """

    def __init__(self,
                 initial_dialog_history=None,
                 agent_type="moderator",
                 engine="gpt-3.5-turbo",
                 api_key="",
                 trace_n_history=2,
                 ):
        """Initialize the moderator agent"""
        super().__init__(initial_dialog_history=initial_dialog_history,
                         agent_type=agent_type,
                         engine=engine,
                         api_key=api_key
                         )

        self.trace_n_history = trace_n_history
        print("Initializing moderator with engine %s" % self.engine)
        return

    def moderate(self,
                 dialog_history, who_was_last="buyer",
                 retry=True):
        """Moderate the conversation between the buyer and the seller"""
        history_len = len(dialog_history)
        if (who_was_last == "buyer"):
            prompt = "buyer: %s\n" % dialog_history[history_len - 1]["content"]
            offset = 1
        else:
            prompt = "seller: %s\n" % dialog_history[history_len - 1]["content"]
            offset = 0

        for i in range(self.trace_n_history - 1):
            idx = history_len - i - 2
            content = dialog_history[idx]["content"]
            if (i % 2 == offset):
                prompt = "buyer: %s\n" % content + prompt
            else:
                prompt = "seller: %s\n" % content + prompt

        prompt += "question: have the seller and the buyer achieved a deal? Yes or No\nanswer:"
        self.last_prompt = prompt

        messages = deepcopy(self.dialog_history)
        messages[-1]['content'] += "\n\n" + prompt

        response = self.call_engine(messages)
        return response['content']


class SellerCriticAgent(DialogAgent):

    def __init__(self,
                 initial_dialog_history=None,
                 agent_type="critic",
                 engine="gpt-3.5-turbo",
                 api_key="",
                 expertise="lobbyist",
                 ):
        """Initialize the seller critic agent"""
        super().__init__(initial_dialog_history=initial_dialog_history,
                         agent_type=agent_type,
                         engine=engine,
                         api_key=api_key
                         )

        print("Initializing seller critic with engine %s" % self.engine)
        return

    def criticize(self, seller_history):
        """Criticize the seller's negotiation strategy"""
        prompt = "\n"
        for d in seller_history[1:]:
            if (d["role"] == "user"):
                prompt += "buyer: %s\n" % d["content"]
            elif (d["role"] == "assistant"):
                prompt += "seller: %s\n" % d["content"]
        prompt += "\n\nNow give three suggestions to improve the seller's negotiation strategy: "

        # TODO: store the history of the critic
        messages = deepcopy(self.dialog_history)
        messages[-1]['content'] += "\n\n" + prompt

        response = self.call_engine(messages)
        feedback = response['content'].replace('\n\n', '\n')
        return feedback


class BuyerCriticAgent(DialogAgent):

    def __init__(self,
                 initial_dialog_history=None,
                 agent_type="critic",
                 engine="gpt-3.5-turbo",
                 api_key="",
                 ):
        """Initialize the buyer critic agent"""
        super().__init__(initial_dialog_history=initial_dialog_history,
                         agent_type=agent_type,
                         engine=engine,
                         api_key=api_key
                         )

        print("Initializing buyer critic with engine %s" % self.engine)
        return

    def criticize(self, buyer_history):
        prompt = "\n"
        for d in buyer_history[1:]:
            if (d["role"] == "user"):
                prompt += "seller: %s\n" % d["content"]
            elif (d["role"] == "assistant"):
                prompt += "buyer: %s\n" % d["content"]
        prompt += "\n\nNow give three suggestions to improve the buyer's negotiation strategy: "

        messages = deepcopy(self.dialog_history)
        messages[-1]['content'] += "\n\n" + prompt

        response = self.call_engine(messages)
        feedback = response['content'].replace('\n\n', '\n')
        return feedback
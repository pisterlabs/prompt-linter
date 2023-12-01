# %%
import json
import os
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from langchain.llms import Anthropic
from pydantic import BaseModel
import xml.etree.ElementTree as ET

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# %%


class User(BaseModel):
    handle: str
    name: str
    bio: str


class ActionType(str, Enum):
    COMMENT = "COMMENT"
    LIKE = "LIKE"
    RETWEET = "RETWEET"
    QUOTE = "QUOTE"
    TWEET = "TWEET"


class Tweet(BaseModel):
    name: str  # this is a user's name
    content: str  # content of the tweet


class Action(BaseModel):
    type: ActionType
    content: Optional[str]
    target: Optional[Tweet]

    def __str__(self):
        """Return action as XML."""
        if self.type == ActionType.TWEET:
            return f"""\
<tweet>
    {self.content}
</tweet>\
"""
        elif self.type == ActionType.COMMENT:
            return f"""\
<comment>
    <parent author="{self.target.name}">
        {self.target.content}
    </parent>
    {self.content}
</comment>\
"""
        elif self.type == ActionType.LIKE:
            return f"""\
<like>
    <parent author="{self.target.name}">
        {self.target.content}
    </parent>
</like>\
"""
        elif self.type == ActionType.RETWEET:
            return f"""\
<retweet>
    <parent author="{self.target.name}">
        {self.target.content}
    </parent>
</retweet>\
"""
        elif self.type == ActionType.QUOTE:
            return f"""\
<quote>
    <parent author="{self.target.name}">
        {self.target.content}
    </parent>
    {self.content}
</quote>\
"""
        else:
            raise ValueError(f"Invalid action type: {type}")


class UserData(BaseModel):
    handle: str
    name: str
    bio: str
    activity: list[Action]


# %%
llm = Anthropic(
    anthropic_api_key=ANTHROPIC_API_KEY,
    model="claude-2",
    temperature=0.8,
)

# %%

sample = UserData.parse_file("samples/jess.json")
print(sample)


# %%


def build_prompt(user: UserData) -> str:
    prompt = "\n\nHuman: "
    prompt += f'You are a Twitter user named {user.name}. Your bio is: "{user.bio}". Below is a collection of your past activity, formatted as XML:\n'
    prompt += "<activity>\n"
    for action in user.activity:
        prompt += str(action)
        prompt += "\n"

    prompt += "</activity>"

    prompt += "Based on this activity, generate 3 new actions that you might take during your next Twitter session. Return your response as XML, matching the schema from above. Only generate tweets, comments, likes, retweets, and quotes."
    prompt += "\n\nAssistant:"

    prompt += " Here are 3 new actions I might take during my next Twitter session:\n<activity>\n"

    return prompt


prompt = build_prompt(sample)
print("PROMPT\n" + prompt)
result = llm.generate([prompt])
result_text = result.generations[0][0].text

# result_text = "[\n" + result_text
# result_text = result_text.strip("```")
print("RESULT TEXT\n" + result_text)
# print(json.loads(result_text))

# %%
json.loads(result_text)

# %%

import xml.etree.ElementTree as ET

def new_like():
    pass

def new_comment():
    pass

def new_retweet():
    pass

def new_quote():
    pass

def new_tweet():
    pass

def parse_xml(xml_text):
    root = ET.fromstring(xml_text)
    actions = []

    for action in root:
        if action.tag == 'tweet':
            actions.append(Action(type=ActionType.TWEET, content=action.text.strip(), target=None))

        elif action.tag == 'retweet':
            parent = action.find('parent')
            tweet = Tweet(name=parent.get('author'), content=parent.text.strip())
            actions.append(Action(type=ActionType.RETWEET, content=None, target=tweet))

        elif action.tag == 'quote':
            parent = action.find('parent')
            tweet = Tweet(name=parent.get('author'), content=parent.text.strip())
            content = ''.join(action.itertext()).strip()  # itertext() will get all text inside 'quote', including 'parent' text
            content = content[len(tweet.content):].strip()  # strip out the 'parent' text to get 'quote' text
            actions.append(Action(type=ActionType.QUOTE, content=content, target=tweet))

    return actions

result_actions = parse_xml("\n<activity>\n"+result_text)
for action in result_actions:
    print(action)
# %%

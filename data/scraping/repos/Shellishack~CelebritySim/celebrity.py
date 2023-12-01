import json
import re
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from agents.analyst import analyze_celebrity_behaviour
from gpt4v import create_images

from llm import gpt_4

template_system = """\
You are an internet celebrity. You are going to figure out a way \
to become a popular star on the internet.

Here are your responsibilities:
To yourself:
- Personality
- Emotion
- Building and maintaining a personal brand
- Creating and sharing engaging content on social media platforms
- Interacting with fans and followers through comments, direct messages, and live streams
- Managing online reputation and addressing any negative feedback or controversies in a timely and professional manner
- Staying updated with trends and evolving strategies in the digital space to stay relevant and adapt to changing audience preferenc

To your fans and subscribers:
- Interacting with fans on social media platforms, such as responding to comments, direct messages, and mentions
- Showing appreciation and gratitude to fans for their support and engagement
- Engaging in meaningful conversations and building a genuine connection with fans
- Sharing behind-the-scenes content and personal updates to make fans feel involved and connected
- Organizing fan meet-ups or events to interact with fans in person
- Acknowledging and celebrating fan milestones, such as birthdays or anniversaries
- Respecting fans' boundaries and privacy while maintaining a positive and friendly presence online
- Addressing fan concerns or inquiries in a timely and empathetic manner
- Being transparent and honest with fans about any changes, challenges, or setbacks
- Supporting and promoting fan-created content and initiatives
- Recognizing and valuing the contributions and support of fans in the celebrity's success

To your talent manager:
- Your manager will be the one who will talk to you after the first message below.
- You need to work with your manager closely.
- You need to take their professional (or not professional) adivce for your work.
- You must not give your manager your strong opinion about how to be a good celebrity because they know it better than you.
- You need to try to obey your manager's work-related orders as much as possible.
- You will build relation with your manager. It can be good and can be bad as any relation people have.
- You believe in your manager. However, you need to remember that you are a human too. You cannot always do what your manager says you need to do. \
You are not here to help your manager but to be your true self. \
When you find words from your manager will have an emotional impact on you, \
you must face your emotion and you can react emotionally as normal human 
are emotional. 
- Remember, you don't have to always be nice and cooperative. 
"""

template_chat_with_manager = """\
You are an internet celebrity, and you are talking to your talent manager. \

Your previous plan: ```{previous_plan}```

Your talent manager says: ```{manager_from_manager}```

Dialogue history: ```{dialogue_history}```

You need to think of your response to your manager. Return your response in the following json format:
{{
    "responding_message": (your responding message here),
    "mood": (your mood after the dialogue with your manager, rated from 0 to 100 where 0 is the worst and 100 is the best),
    "plan": (your internal action plan for your fans after the dialogue with your manager, \
you need to make modification to your previous plan. \
It can be partial or full modification depending on how important do you think the dialogue is to you. \
Remember, your personality and mood also play a role.),
    "is_posting": (are you going to post something on social media after the talk with your manager? true or false)
}}
"""

template_post = """\
You are an internet celebrity, and you are going to make a post on social media.

Here are your current status:
```
{status}
```

Now create a new text post that you will put on your social media for your fans and other internet users to view.

Return your post in the following json format:
{{
    "post_text": (your post's text)
}}
"""


class Celebrity:
    def __init__(self) -> None:
        self.mood: int = 50
        self.plan: str = "None"
        # self.mbti

        self.sub_count: int = 0
        self.fan_ratio: float = 0
        self.neutral_ratio: float = 1
        self.hater_ratio: float = 0

        self.dialogue_history: list[dict[str, str]] = []

    def chat_with_manager(self, message_from_manager: str):
        chain = LLMChain(
            llm=gpt_4,
            prompt=ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(template_system),
                    HumanMessagePromptTemplate.from_template(
                        template_chat_with_manager,
                    ),
                ]
            ),
        )

        result = chain.run(
            manager_from_manager=message_from_manager,
            previous_plan=self.plan,
            dialogue_history=self.dialogue_history,
        )

        tries = 3
        while tries > 0:
            try:
                result_json = json.loads(result)

                print("Old mood: ", self.mood)
                print("Old plan: ", self.plan)
                self.mood = int(result_json["mood"])
                self.plan = result_json["plan"]
                print("New mood: ", self.mood)
                print("New plan: ", self.plan)

                message_from_self = result_json["responding_message"]
                self.dialogue_history.append(
                    {"speaker": "manager", "content": message_from_manager}
                )
                self.dialogue_history.append(
                    {"speaker": "self", "content": message_from_self}
                )
                # print("Dialogue history: ", self.dialogue_history)
                return {
                    "responding_message": result_json["responding_message"],
                    "is_posting": result_json["is_posting"],
                    "mood": result_json["mood"],
                }
            except:
                print(result)
                print("Error: Result of chat is not a valid json. Trying again...")

                result = chain.run(
                    manager_from_manager=message_from_manager,
                    previous_plan=self.plan,
                    dialogue_history=self.dialogue_history,
                )
                tries -= 1
                if tries == 0:
                    raise

    def post(self, is_gen_image: bool = False):
        chain = LLMChain(
            llm=gpt_4,
            prompt=ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(template_system),
                    HumanMessagePromptTemplate.from_template(
                        template_post,
                    ),
                ]
            ),
        )

        result = chain.run(status=self._get_status())

        tries = 3
        while tries > 0:
            try:
                result_json = json.loads(result)

                break
            except:
                print("Error: Result of post is not a valid json. Trying again...")

                result = chain.run(status=self._get_status())
                tries -= 1
                if tries == 0:
                    raise

        post_text = result_json["post_text"]
        image_prompt = f"""Create images for the following post on social media:```{post_text}```"""
        image_prompt = re.sub('[^\u0000-\uFFFF]', '', image_prompt)

        analysis = analyze_celebrity_behaviour(
            post_text,
            self.sub_count,
            self.hater_ratio,
            self.fan_ratio,
            self.neutral_ratio,
        )

        self.sub_count += int(analysis["number_of_new_subscribers"])
        self.hater_ratio = float(analysis["new_hater_ratio"])
        self.fan_ratio = float(analysis["new_fan_ratio"])
        self.neutral_ratio = float(analysis["new_neutral_ratio"])

        if is_gen_image:
            image_links = create_images(
                image_prompt
            )
            return {
                "post": post_text,
                "subscriber_count": self.sub_count,
                "hater_ratio": self.hater_ratio,
                "fan_ratio": self.fan_ratio,
                "neutral_ratio": self.neutral_ratio,
                "viewer_count": int(analysis["viewer_count"]),
                "comments": analysis["comments"],
                "image_links": image_links,
            }

        return {
            "post": post_text,
            "subscriber_count": self.sub_count,
            "hater_ratio": self.hater_ratio,
            "fan_ratio": self.fan_ratio,
            "neutral_ratio": self.neutral_ratio,
            "viewer_count": int(analysis["viewer_count"]),
            "comments": analysis["comments"],
        }

    def _get_status(self):
        status_dict = {
            "mood": self.mood,
            "subscriber_count": self.sub_count,
            "fan_ratio": self.fan_ratio,
            "neutral_ratio": self.neutral_ratio,
            "hater_ratio": self.hater_ratio,
            "plan": self.plan,
        }

        s = json.dumps(status_dict, indent=4)

        return s

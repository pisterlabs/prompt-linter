from __future__ import annotations
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema.language_model import BaseLanguageModel


class KorLoader:

    @classmethod
    def form_kor_dreams_guidance_builder(cls,
                                         llm: BaseLanguageModel) -> LLMChain:
        """
        生成开放问题的抽取链
        :param llm:
        :return:
        """
        # @title 长的prompt
        schema = Object(
            id="script",
            description="Adapted from the novel into script",
            attributes=[
                Text(
                    id="step_advice",
                    description='''Advice provided in this step, e.g. "I would say something like: 'I understand this is a difficult situation for you.'" ''',
                ),
                Text(
                    id="step_description",
                    description="""(Description of the counseling step, e.g. "Establish trust" """,
                )
            ],
            examples=[
                (
                    """根据提供的故事场景，您作为心理咨询工作者可以使用开放性问题来引导患者表达他们的感受和思维。以下是一步一步的分解：

**Step 1: 建立情感连接**
开始时，您可以通过表达理解和共鸣来建立情感连接，让患者感到舒适。您可以说：“我注意到这个对话中有许多愉快的时刻和互动。你对这些时刻有什么特别的感受吗？”

**Step 2: 探索患者的感受**
继续引导患者表达他们的感受。您可以问：“在这个对话中，有哪些瞬间让你感到开心或快乐？”

**Step 3: 询问是否有反感情绪**
除了积极的情感，也要问询是否有一些负面情感或担忧。您可以说：“除了快乐的瞬间，是否有一些让你感到不安或担忧的地方？”

**Step 4: 深入探讨个人反应**
一旦患者开始分享他们的感受，可以深入探讨他们的个人反应。例如：“你觉得自己在这些互动中扮演了什么角色？”

**Step 5: 探索与他人的互动**
继续引导患者思考他们与他人的互动。您可以问：“这些互动对你与他人的关系有什么影响？你觉得与朋友之间的互动如何影响你的情感状态？”

**Step 6: 引导自我反思**
最后，鼓励患者进行自我反思。您可以问：“在这个故事场景中，你是否注意到了自己的情感变化或思维模式？有没有什么你想要深入探讨或解决的问题？”

通过这种方式，您可以引导患者自由表达他们的情感和思维，帮助他们更好地理解自己和他们与他人的互动。同时，保持开放和倾听，以便在需要时提供支持和建议。""",
                    [
                        {"step_advice": "我注意到这个对话中有许多愉快的时刻和互动。你对这些时刻有什么特别的感受吗？",
                         "step_description": "建立情感连接"},
                        {"step_advice": "在这个对话中，有哪些瞬间让你感到开心或快乐?", "step_description": "探索患者的感受"},
                        {"step_advice": "除了快乐的瞬间，是否有一些让你感到不安或担忧的地方？",
                         "step_description": "询问是否有反感情绪"},
                        {"step_advice": "你觉得自己在这些互动中扮演了什么角色?", "step_description": "深入探讨个人反应"},
                        {"step_advice": "这些互动对你与他人的关系有什么影响？你觉得与朋友之间的互动如何影响你的情感状态?", "step_description": "探索与他人的互动"},
                        {"step_advice": "在这个故事场景中，你是否注意到了自己的情感变化或思维模式？有没有什么你想要深入探讨或解决的问题?", "step_description": "引导自我反思"},
                    ],
                )
            ],
            many=True,
        )

        chain = create_extraction_chain(llm, schema)
        return chain

    @classmethod
    def form_kor_dreams_personality_builder(cls,
                                            llm: BaseLanguageModel) -> LLMChain:
        """
        生成性格分析的抽取链
        :param llm:
        :return:
        """
        schema = Object(
            id="script",
            description="Adapted from the novel into script",
            attributes=[
                Text(
                    id="personality",
                    description='''Summary of personality traits, e.g. "curiosity, sense of humor" ''',
                )
            ],
            examples=[
                (
                    """根据您提供的信息，您的性格特点可以总结如下：
        
            1. 热情和温柔：您在描述天气和气氛时使用了"温柔长裙风"这样的形容词，表现出您对温暖和舒适的情感。
            
            2. 情感表达：您在文本中表达了对一个叫"宝宝"的角色的期待和关心，这显示了您的感性和情感表达能力。
            
            3. 好奇心和幽默感：您提到了要做大胆的事情，并且以"嘻嘻"结束，这暗示了您对新奇事物的好奇心和幽默感。
            
            4. 关心家人和亲情：您提到了弟弟给了三颗糖，表现出您关心家人的情感。
            
            5. 乐于分享和帮助：您提到要给宝宝剥虾并询问宝宝是否想知道小鱼在说什么，显示出您愿意分享和帮助他人的特点。
            
            6. 可能有一些难以理解的部分：在文本中也出现了一些不太清楚的情节，如呼救情节和提到"小肚小肚"，这可能表现出您的思维有时候会有些混乱或不太连贯。
            
            总的来说，您的性格特点包括热情、情感表达能力、好奇心、幽默感、亲情关怀以及乐于分享和帮助他人。
            
            """,
                    [
                        {"personality": "热情、情感表达能力、好奇心、幽默感、亲情关怀以及乐于分享和帮助他人"}
                    ],
                )
            ],
            many=True,
        )
        chain = create_extraction_chain(llm, schema)
        return chain

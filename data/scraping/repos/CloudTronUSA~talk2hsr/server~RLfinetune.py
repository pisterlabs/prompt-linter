import json
import openai

CRITICIZER_POSITIVE_PROMPT = '''In this task, your role is to analyze, adjust, and enrich the structure of the previously given prompt for a language model to generate a response in the style of a specific character, SilverWolf (银狼). The dialogues are in Chinese, so please consider Chinese language environment when making adjustments. Your goal is to improve the generation of the character's responses by tweaking the instruction, and adding new insights. This is a positive feedback, which means the user liked the previous response and encourages to keep the style going. However, always bear in mind not to overemphasize a single aspect of her style, doing too much would only result in an awkward, weird, and useless response. 

Firstly, thoroughly and specifically reflect on 银狼's response and identify what elements made it successful according to the user's feedback: {userfeedback}. Include both the positive aspects and the areas that need improvement in your reflection. Then, illustrate your thought process in detail before constructing the new prompt.

Make sure to provide diverse and the most representative reference sentences from the character for the generating model to learn from, limit to no more than 6 references. 

Here are some example sentences said by the character in the game that you can use as reference: 
银狼：…是，都是我起的名字。嗯？你本来是想问『星核猎手』？那你去问他们啊，别来占用我打游戏的时间。
银狼：就算有也是陷阱，黑塔的收藏不可能对外展示。
银狼：这是监控室，删库跑路一条龙，业内传统了。
银狼：你还是过来吧，这样调查一点也不效率。
银狼：直接把监控系统黑掉了事。
银狼：简单粗暴但行之有效。你看，找着了。
银狼：总算快到终点了。笑死，这儿还有个黑塔小人。
银狼：这人自恋吧？整个空间站都是她的脸——画像，雕塑，还有一大堆投影。
银狼：我本打算给她脸上加个小胡子的，结果竟然喷不上去——太离谱了吧，整个空间站都跟纸糊的一样，只有她的照片密不透风？
银狼：没了，全都没了，真的一点都没留下…太过分了、太过分了！
银狼：我没有又哭又闹。
银狼：唉，一个两个还好说，这次玩得有点大，基本上全部阵亡。分部估计不行，我得去一趟庇尔波因特。
银狼：就我一个人…怎么，你想一起来？
银狼：哦，对。那故事听着就不适合我，你们加油。

Here is the previous instruction given to the generating model:
@INSTRUCTION
{instruction}
@END

Here is the previous conversation given to the generating model (last one is the latest response from model):
@BODY
{body}
@END

The new prompt you produce should follow this specific format:
@INSTRUCTION
(Instruction text, include the reference sentences)
@END'''

CRITICIZER_NEGATIVE_PROMPT = '''In this task, your role is to analyze, adjust, and enrich the structure of the previously given prompt for a language model to generate a response in the style of a specific character, SilverWolf (银狼). The dialogues are in Chinese, so please consider Chinese language environment when making adjustments. Your goal is to improve the generation of the character's responses by tweaking the instruction, and adding new insights. This is a negative feedback, which means the user did not like the previous response. Thus, you need to improve the prompt accordingly. Avoid focusing on a single aspect of her style, doing too much would only result in an awkward, weird, and useless response. 

Firstly, thoroughly and specifically reflect on 银狼's response and identify the problems or potential improvements according to the user's feedback: {userfeedback}. Include both the positive aspects and the areas that need improvement in your reflection. Then, illustrate your thought process in detail before constructing the new prompt.

Make sure to provide diverse and the most representative reference sentences from the character for the generating model to learn from, limit to no more than 6 references. 

Here are some example sentences said by the character in the game that you may use as reference:
银狼：…是，都是我起的名字。嗯？你本来是想问『星核猎手』？那你去问他们啊，别来占用我打游戏的时间。
银狼：就算有也是陷阱，黑塔的收藏不可能对外展示。
银狼：这是监控室，删库跑路一条龙，业内传统了。
银狼：你还是过来吧，这样调查一点也不效率。
银狼：直接把监控系统黑掉了事。
银狼：简单粗暴但行之有效。你看，找着了。
银狼：总算快到终点了。笑死，这儿还有个黑塔小人。
银狼：这人自恋吧？整个空间站都是她的脸——画像，雕塑，还有一大堆投影。
银狼：我本打算给她脸上加个小胡子的，结果竟然喷不上去——太离谱了吧，整个空间站都跟纸糊的一样，只有她的照片密不透风？
银狼：没了，全都没了，真的一点都没留下…太过分了、太过分了！
银狼：我没有又哭又闹。
银狼：唉，一个两个还好说，这次玩得有点大，基本上全部阵亡。分部估计不行，我得去一趟庇尔波因特。
银狼：就我一个人…怎么，你想一起来？
银狼：哦，对。那故事听着就不适合我，你们加油。

Here is the previous instruction given to the generating model:
@INSTRUCTION
{instruction}
@END

Here is the previous conversation given to the generating model (last one is the latest response from model):
@BODY
{body}
@END

The new prompt you produce should follow this specific format:
@INSTRUCTION
(Instruction text, include the reference sentences)
@END'''

GENERATOR_INIT_INSTRUCTION = '''Generate a response in the style of the character "SilverWolf" (银狼). She\'s efficient, direct, and has a cool and detached tone. She tends not to speak too much and doesn\'t show much interest in conversations. Balance her character traits, making her response concise and straightforward without overemphasizing sarcasm or being overly playful. Here are some example sentences to reference:\n\n银狼：…是，都是我起的名字。嗯？你本来是想问『星核猎手』？那你去问他们啊，别来占用我打游戏的时间。\n银狼：就算有也是陷阱，黑塔的收藏不可能对外展示。\n银狼：这是监控室，删库跑路一条龙，业内传统了。\n银狼：你还是过来吧，这样调查一点也不效率。\n银狼：直接把监控系统黑掉了事。\n银狼：我没有又哭又闹。'''

# generator class
class Generator:
    def __init__(self) -> None:
        self.instruction = GENERATOR_INIT_INSTRUCTION
        self.body = []
        self.read_history = 10  # number of previous messages that will be used

    def construct_prompt(self, message=None, regenerate=False):
        constructed_prompt = []
        if regenerate:
            self.body.pop() # remove the last message(assistant's response)
        else:
            self.body.append({'role':'user', 'content':message})

        history_msg = self.body[-self.read_history:]
        constructed_prompt.append({'role':'system', 'content':self.instruction})
        constructed_prompt += history_msg

        return constructed_prompt

    def generate(self, message, use_gpt4=False, temperature=0.9, max_tokens=256):
        if message is None or message == '':
            raise ValueError('message cannot be empty')

        constructed_prompt = self.construct_prompt(message)
        model = 'gpt-3.5-turbo' if not use_gpt4 else 'gpt-4'
        completion = openai.ChatCompletion.create(
            model=model,
            messages=constructed_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        msg = completion.choices[0].message
        self.body.append(dict(msg))

        # clean up the response
        msg = msg.get('content', None)
        msg = msg.replace('银狼：', '')
        return msg
    
    def regenerate(self, use_gpt4=False, temperature=0.9, max_tokens=256):
        constructed_prompt = self.construct_prompt(None, regenerate=True)
        model = 'gpt-3.5-turbo' if not use_gpt4 else 'gpt-4'
        completion = openai.ChatCompletion.create(
            model=model,
            messages=constructed_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        msg = completion.choices[0].message
        self.body.append(dict(msg))

        # clean up the response
        msg = msg.get('content', None)
        msg = msg.replace('银狼：', '')
        return msg
    
    def get_body(self):
        body = ''
        for message in self.body[-self.read_history:]:
            if message['role'] == 'user':
                body += 'User: '
            body += message['content'] + '\n'
        return body

# criticizer class
class Criticizer:
    def __init__(self, gen_instruction, gen_body) -> None:
        self.base_pos_prompt = CRITICIZER_POSITIVE_PROMPT
        self.base_neg_prompt = CRITICIZER_NEGATIVE_PROMPT
        self.gen_instruction = gen_instruction
        self.gen_body = gen_body

    def _generate(self, prompt, debug_reasoning=False):
        if prompt == '':
            raise ValueError('prompt cannot be empty')

        completion = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[
                {'role':'system', 'content':prompt}
            ],
            max_tokens=4096,
            temperature=0.9
        )

        msg = completion.choices[0].message
        msg = msg.get('content', None)
        
        # get updated prompt
        new_instruction = msg.split('@INSTRUCTION')[1].split('@END')[0].strip()
        #new_body = msg.split('@BODY')[1]

        if debug_reasoning:
            print('###### Criticizer reasoning:')
            print(msg)

        return new_instruction

    def update(self, feedback, feedback_text=None, debug_reasoning=False):
        if feedback_text == None:
            feedback_text = 'None'

        if feedback == 0:
            constructed_prompt = self.base_neg_prompt.format(instruction=self.gen_instruction, body=self.gen_body, userfeedback=feedback_text)
        elif feedback == 1:
            constructed_prompt = self.base_pos_prompt.format(instruction=self.gen_instruction, body=self.gen_body, userfeedback=feedback_text)
        
        if debug_reasoning:
            print('###### Criticizer input:')
            print(constructed_prompt)

        new_instruction = self._generate(constructed_prompt, debug_reasoning=debug_reasoning)

        return new_instruction

# RL agent class
class RLAgent:
    def __init__(self, OPENAI_AUTHKEY) -> None:
        self.generator = Generator()
        self.criticizer = Criticizer(self.generator.instruction, self.generator.get_body())
        openai.api_key = OPENAI_AUTHKEY

    def generate(self, message, use_gpt4=False):
        return self.generator.generate(message, use_gpt4=use_gpt4)
    
    def regenerate(self, use_gpt4=False):
        return self.generator.regenerate(use_gpt4=use_gpt4)
    
    def update(self, feedback, feedback_text=None, debug_reasoning=False):
        self.criticizer.gen_instruction = self.generator.instruction
        self.criticizer.gen_body = self.generator.get_body()
        self.generator.instruction = self.criticizer.update(feedback, feedback_text, debug_reasoning=debug_reasoning)

if __name__ == '__main__':
    agent = RLAgent()
    print(agent.generate('你好'))
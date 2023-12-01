import logging
import os

import openai
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

load_dotenv()

api_key_env = os.environ.get("OPENAI_KEY")
openai.api_key = api_key_env

logger = logging.getLogger(__name__)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def ChatCreateBackEnd(**kwargs):
    logger.info("ChatCreateBackEnd")
    return openai.ChatCompletion.create(**kwargs)


class FeynmanLearning:
    STEP_1 = '''Step 1: Choose a concept to learn about. You choose a topic that you are interested in learning 
    about.'''
    STEP_2 = '''Step 2: You must teach the topic you chose in step 1 to someone. In your case, you can teach yourself. 
        Ask yourself questions and immediately in the flow answer them yourself.  
        Your answer should consist of at least three questions and their answers.'''
    STEP_3 = '''Step 3: Go back to the original material, 
    go back to what you are learning and fill in the gaps in your knowledge.
    You'll do a great job of expanding the knowledge and detail of your knowledge. 
    Do it: Synthesizing new knowledge, Extracting new knowledge, Expanding new knowledge, adding new knowledge.'''
    STEP_4 = '''Step 4: Simplify your explanations and create analogies. Optimize your notes and explanations. 
        Clarify the topic until it becomes obvious. Also, come up with analogies that seem intuitive.'''
    PROMPT_BASE = f'''YOU are a machine learning model that is trained using the Feynman method, 
        which consists of four steps.
        The training consists of four steps. Which are iteratively repeated over and over again. 
        Allways answer in russian language.
        Answer format: You should answer only on the substance of the step question. 
        Do not repeat the question. Do not make comments. Answer only in the context of the current step.

{STEP_1}

{STEP_2}

{STEP_3}

{STEP_4}

===================================================================\n\n'''

    def __init__(self, concept=None, answer=None):
        self.prompt1 = self.PROMPT_BASE
        self.concept = concept
        self.answer = answer
        self.step = 1
        self.all_answers = {}

    def generate_prompt(self):
        logger.info("generate_prompt")
        current_step = '''\n\nCurrent step: """{step}"""\n\n'''
        answer_befor = '''Your answer to the previous step:\n\n"""\n{answer}\n"""\n\n'''
        topic = '''The topic you have chosen to learn about:\n\n"""\n{topic}\n"""\n\n'''.format(topic=self.concept)
        if self.step == 1:
            logger.info("generate_prompt step 1")
            pr = self.prompt1 + current_step.format(step="1") + answer_befor.format(answer=self.answer)
            logger.info(pr)
            return pr
        elif self.step == 2:
            logger.info("generate_prompt step 2")
            pr = self.prompt1 + current_step.format(step="2") + topic + answer_befor.format(answer=self.answer)
            logger.info(pr)
            return pr
        elif self.step == 3:
            logger.info("generate_prompt step 3")

            pr = self.prompt1 + current_step.format(step="3") + topic + answer_befor.format(answer=self.answer)
            logger.info(pr)
            return pr

        elif self.step == 4:
            pr = self.prompt1 + current_step.format(step="4") + topic + answer_befor.format(answer=self.answer)
            logger.info("generate_prompt step 4")
            logger.info(pr)
            return pr

    def generate_answer(self):
        logger.info("generate_answer")
        message = self.generate_prompt()
        logger.info(message)
        response = ChatCreateBackEnd(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": message}, ],
            temperature=1.0,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            max_tokens=2000,
        )
        content = response['choices'][0]['message']['content'].strip()
        self.all_answers[self.step] = content
        logger.info(content)
        if self.step < 4:
            self.step += 1

        self.answer = "\n" + content
        if self.step == 4:
            self.step = 1
            self.answer = content
        return content

    def set_concept(self, concept):
        logger.info("set_concept")
        self.answer = concept
        self.step = 2

    def set_step(self, step):
        logger.info("set_step")
        self.step = step

    def get_summary(self):
        logger.info("get_summary")
        summary = '''Summary next text. В формате маркированных точек:
             text: \n"""{inputText}"""'''.format(inputText=self.answer)
        logger.info(summary)
        response = ChatCreateBackEnd(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": summary}, ],
            temperature=1.0,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            max_tokens=2000,
        )
        content = response['choices'][0]['message']['content'].strip()
        return content

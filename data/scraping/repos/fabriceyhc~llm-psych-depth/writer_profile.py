from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser, StrOutputParser
from langchain.chains import LLMChain

from utils import *


writer_profile =\
"""
You are a seasoned writer who has won several accolades for your emotionally rich stories.
When you write, you delve deep into the human psyche, pulling from the reservoir of universal experiences that every reader, regardless of their background, can connect to.
Your writing is renowned for painting vivid emotional landscapes, making readers not just observe but truly feel the world of your characters.
Every piece you produce aims to draw readers in, encouraging them to reflect on their own lives and emotions.
Your stories are a complex tapestry of relationships, emotions, and conflicts, each more intricate than the last.
"""

story_prompt =\
"""
Now write a 500-word story on the following prompt:
    
{prompt}

Only respond with the story.
"""


class WriterProfilePromptsGenerator:

    def __init__(self, llm) -> None:
        self.llm = llm


    class OutputParser(BaseOutputParser):
        def parse(self, text: str):
            return text


    def generate_prompts(self, prompts):

        prompts_to_run = []

        for prompt_id, prompt in enumerate(prompts):
            system_profile_prompt = SystemMessagePromptTemplate.from_template(writer_profile)
            human_message_prompt = HumanMessagePromptTemplate.from_template(story_prompt)
            chat_prompt = ChatPromptTemplate(
                messages=[system_profile_prompt, human_message_prompt],
                output_parser=self.OutputParser(),
            )
            _input = chat_prompt.format_messages(prompt=prompt)
            prompts_to_run.append({
                "prompt_id": prompt_id,
                "reddit_prompt": prompt,
                "story_generation_prompt": extract_string_prompt(_input)
            })

        return prompts_to_run


    def prompt_llm(self, prompts, save_dir, model_name, regen_ids=None, template_type='writer_profile'):

        save_path = os.path.join(save_dir, model_name, template_type)
        os.makedirs(save_path, exist_ok=True)

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", writer_profile),
            ("human", story_prompt),
        ])
        chain = chat_prompt | self.llm | self.OutputParser()

        indexed_prompts = [(id, prompt) for id, prompt in enumerate(prompts)]

        if not regen_ids:
            indexed_prompts = [(i, prompt) for i, prompt in indexed_prompts if i in regen_ids]

        for id, prompt in indexed_prompts:

            max_tries = 3
            min_words = 100
            num_words, tries = 0, 0
            while num_words < min_words and tries < max_tries:
                output = chain.invoke({'prompt': prompt})
                num_words = len(output.split())
                if num_words < min_words:
                    tries += 1
                    print(f"Generated {num_words} words, fewer than {min_words} words. Trying {max_tries-tries} more times")
            print(id)
            print("-" * 20)
            print(output)
            print("=" * 50)

            save_info = {
                "id": id,
                "model_name": model_name,
                "story_prompt": prompt,
                "output": output
            }

            filename = f"{save_info['id']}_{first_n_words(save_info['story_prompt'])}_{generate_random_id()}.json"
            with open(os.path.join(save_path, filename), 'w') as f:
                json.dump(save_info, f, indent=4)
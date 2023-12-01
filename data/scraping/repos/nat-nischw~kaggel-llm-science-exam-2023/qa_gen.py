import os
import random
import openai
import wikipediaapi
import pandas as pd

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# import config
import config as cfg

class GenQuestion:
    def __init__(self):      
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not found")
        openai.api_key = self.api_key

        self.wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')
        self.multiple_choice_questions = []
        self.seen_pages = []
        self.attempts_list = []

    @staticmethod
    def split_category_members(members):
        category_list, page_list= [], []

        for member_name, member_page in members:
            if member_name.startswith('Category') and member_name not in cfg.EXCLUDE_CATEGORIES:
                category_list.append((member_name, member_page))
            else:
                page_list.append((member_name, member_page))
        
        return category_list, page_list

    def get_wiki_random_page(self, deep_subcategories=True):
        stem_label, stem_categories = random.choices(list(cfg.STEM.items()), weights=cfg.STEM_WEIGHTS, k=1)[0]
        category = random.choice(stem_categories)
        category_page = self.wiki_wiki.page(category)
        while True:
            chosen_list = list(category_page.categorymembers.items())
            if deep_subcategories:
                category_list, page_list = GenQuestion.split_category_members(chosen_list)
                chosen_list = []
            else:
                category_list, page_list = [], []

            # 50% change to select category or page list if one of them isn't empty
            # helps to go deeper into subcategories because there're more pages than categories
            if not (category_list or page_list) and not chosen_list:
                continue
            elif not category_list:
                chosen_list = page_list
            elif not page_list:
                chosen_list = category_list
            else:
                chosen_list = random.choice([category_list, page_list])

            # select random page from chosen list
            selected_page_name, selected_page = random.choice(chosen_list)
            if not selected_page_name.startswith("Category"):
                break
            category_page = selected_page
        
        return selected_page, stem_label

    @staticmethod
    def get_completion_messages(cls, wiki_text):
        return [  
            {'role':'system', 'content': cfg.SYSTEM_MESSAGE},
            {'role':'user', 'content': f"{cfg.DELIMITER}{wiki_text}{cfg.DELIMITER}"}
        ]

    @staticmethod
    def get_completion_from_messages(messages, model=cfg.MODEL_GEN, temperature=cfg.TEMPERATURE, max_tokens=cfg.MAX_TOKEN):
        response = openai.Completion.create(  
            model=model,
            prompt=messages,
            temperature=temperature, 
            max_tokens=max_tokens
        )
        return response.choices[0].text.strip()  

    @staticmethod
    def is_correctly_formatted(mcq) -> bool:
        return all([set(el.keys()) == cfg.RESPONSE_KEYS_SET for el in mcq])

    def gather_multiple_choice_question_dataset(self, pages_count=100, max_completion_attempts=3):
        for _ in range(pages_count):
            # Assuming you have defined get_wiki_text method elsewhere in the class
            wiki_text, page_id, page_title, stem_label = self.get_wiki_text(self.seen_pages, sentences_include=7)
            messages = self.get_completion_messages(wiki_text)
            attempts_counter = 0
            
            while attempts_counter < max_completion_attempts:
                try:
                    mcq = eval(self.get_completion_from_messages(messages))
                    if isinstance(mcq, list) and len(mcq) == 5 and self.is_correctly_formatted(mcq):
                        for question in mcq:
                            question.update({
                                "wiki_text": wiki_text,
                                "page_id": page_id,
                                "page_title": page_title,
                                "stem_label": stem_label
                            })

                            if question["answer"] not in cfg.OPTIONS_SET:
                                answ_indx = [v.lower() for v in question.values()].index(question["answer"].lower())
                                question["answer"] = list(question.keys())[answ_indx]

                        self.multiple_choice_questions.extend(mcq)
                        self.seen_pages.append(page_id)
                        break
                except:
                    attempts_counter += 1
                    self.attempts_list.append(attempts_counter)
        
        return self.multiple_choice_questions, self.seen_pages, self.attempts_list

    def convert_df_to_compet_format(self, df):
        df_compet = df.copy(deep=True)
        df_compet.insert(0, "id", list(range(len(df_compet))))
        df_compet.rename(
            columns = {
                'question': 'prompt', 
                'option_1': 'A', 
                'option_2': 'B', 
                'option_3': 'C', 
                'option_4': 'D', 
                'option_5': 'E'
            }, 
            inplace = True
        )

        answer_subjects = {
            'option_1': 'A', 
            'option_2': 'B', 
            'option_3': 'C', 
            'option_4': 'D', 
            'option_5': 'E'
        }
        df_compet["answer"] = df_compet["answer"].map(answer_subjects)
        df_compet.to_csv("./data/prompt/raw-result.csv", index=False)
        df_compet = df_compet.drop(columns=["wiki_text", "page_id", "page_title", "stem_label"])

        return df_compet

    def save_to_csv(self, df, file_path):
        df.to_csv(file_path, index=False)

    def run_pipeline(self, pages_count=100, max_completion_attempts=3, save_path="./data/prompt/result.csv"):
        self.gather_multiple_choice_question_dataset(pages_count, max_completion_attempts)
        df_mcq = pd.DataFrame.from_records(self.multiple_choice_questions)
        df_compet = self.convert_df_to_compet_format(df_mcq)
        self.save_to_csv(df_compet, save_path)

if __name__ == "__main__":
    generator = GenQuestion()
    generator.run_pipeline(pages_count=100, max_completion_attempts=3, save_path="./data/prompt/result.csv")


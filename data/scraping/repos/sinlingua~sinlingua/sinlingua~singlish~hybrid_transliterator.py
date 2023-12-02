import os
import openai
import json
import time
from sinlingua.singlish.rulebased_transliterator import RuleBasedTransliterator
from sinlingua.src.singlish_resources import config_data


class HybridTransliterator:
    def __init__(self, api_key: str = None, org_key: str = None, prompt_masking: str = None, prompt_suggestion: str = None):
        # config_file = "config.json"
        # self.json_data = self.__read_json_config(file_path=config_file)
        self.json_data = config_data
        if api_key is not None and org_key is not None:
            self.json_data["api_key"] = api_key
            self.json_data["org_key"] = org_key
        if prompt_masking is not None:
            self.json_data["Prompts"][0]["content"] = prompt_masking
        if prompt_suggestion is not None:
            self.json_data["Prompts"][1]["content"] = prompt_suggestion

    def view_prompt(self, level: int):
        print(self.json_data["Prompts"][level]["content"])

    # @staticmethod
    # def __read_json_config(file_path: str) -> dict:
    #     try:
    #         # Read JSON configuration file and return the data as dictionary
    #         with open(os.path.join(RESOURCE_PATH, file_path), 'r', encoding='utf-8') as json_file:
    #             json_data_c = json.load(json_file)
    #         return json_data_c
    #     except Exception as e:
    #         # Handle exceptions while reading JSON configuration
    #         print(f"Error while reading JSON configuration file '{file_path}': {str(e)}")
    #         return {}

    def __get_gpt_response(self, text: str, level: int, word: str = "") -> str:
        completion = None
        try:
            # Set up API key and organization for OpenAI
            openai.api_key = self.json_data["api_key"]
            openai.organization = self.json_data["org_key"]

            # Create user prompt using provided text and level
            user_prompt = self.json_data["Prompts"][level]["content"].replace("{{masked-sentence}}", text).replace(
                "{{misspelled-word}}", word)

            # Check if the provided text is empty
            if not text.strip():
                raise ValueError("Text is empty. Please provide a valid text string.")

            success = False
            while not success:
                try:
                    # Create a ChatCompletion request to GPT-3
                    completion = openai.ChatCompletion.create(
                        model=self.json_data["model"],
                        messages=[
                            {
                                "role": self.json_data["Prompts"][level]['role'],
                                "content": user_prompt
                            }
                        ],
                        n=1,
                        temperature=self.json_data['temperature'],
                        max_tokens=self.json_data['max_tokens'],
                        top_p=self.json_data['Top_P'],
                        frequency_penalty=self.json_data['Frequency_penalty'],
                        presence_penalty=self.json_data['Presence_penalty']
                    )
                    success = True
                except Exception as e:
                    # Handle exceptions during GPT-3 request
                    sleep_time = 2
                    time.sleep(sleep_time)
                    print("Error:", e)
                    print("Retrying...")

            result = completion.choices[0].message.content
            sleep_time = 2
            time.sleep(sleep_time)  # To avoid rate limit
            return result
        except Exception as e:
            # Handle exceptions during GPT response processing
            print(f"Error in GPT response for text '{text}': {str(e)}")
            return ""

    @staticmethod
    def __remove_duplicates(input_list: list) -> list:
        unique_words = set()
        result = []

        for word in input_list:
            # Remove duplicates from the input list
            if word not in unique_words:
                unique_words.add(word)
                result.append(word)

        return result

    def transliterator(self, text: str) -> str:
        try:
            # Call parent transliterator to get the initial output
            rule_based = RuleBasedTransliterator()
            output = rule_based.transliterator(text=text)

            # Process output sentence by sentence
            paragraphs = output.split('\n')
            merged_para = []

            for paragraph in paragraphs:
                sentences = paragraph.split(".")
                merged_sentence = []

                for sentence in sentences:
                    try:
                        if sentence == "":
                            continue
                        # Get GPT response for level 0
                        gpt_response_1 = self.__get_gpt_response(text=sentence, level=0)
                        dictionary_1 = json.loads(gpt_response_1)

                        primary_list_item = 0  # Initialize the index for the while loop
                        dictionary_list = dictionary_1["word_list"]
                        dictionary_list = self.__remove_duplicates(dictionary_list)

                        while primary_list_item < len(dictionary_1["word_list"]):
                            for list_item in dictionary_list:

                                # Check for multiple occurrences of the same word
                                if sentence.count(list_item) > 1:
                                    sentence = sentence.replace(list_item, "<mask>", 1)
                                    dictionary_1["word_list"].append(list_item)
                                    dictionary_list.remove(list_item)
                                    dictionary_list.append(list_item)

                                    # Get GPT response for level 1
                                    gpt_response_2 = self.__get_gpt_response(text=sentence, level=1, word=list_item)
                                    dictionary_2 = json.loads(gpt_response_2)

                                    if list_item in dictionary_2:
                                        sentence = sentence.replace("<mask>", dictionary_2[list_item])
                                    else:
                                        sentence = sentence.replace("<mask>", list_item)
                                    break

                                # Check for single occurrence of the word
                                elif sentence.count(list_item) == 1:
                                    sentence = sentence.replace(list_item, "<mask>", 1)
                                    gpt_response_2 = self.__get_gpt_response(text=sentence, level=1, word=list_item)
                                    dictionary_2 = json.loads(gpt_response_2)

                                    if list_item in dictionary_2:
                                        sentence = sentence.replace("<mask>", dictionary_2[list_item])
                                    else:
                                        sentence = sentence.replace("<mask>", list_item)
                                    break

                            primary_list_item += 1

                        merged_sentence.append(sentence)
                    except Exception as e:
                        print(f"Error processing sentence: {str(e)}")
                para = ". ".join(merged_sentence)
                merged_para.append(para)
            full_text = "\n".join(merged_para)
            return full_text

        except Exception as e:
            print(f"Error in transliterator: {str(e)}")
            return text

    def machine_mask(self, text: str,) -> str:
        try:
            # Call parent transliterator to get the initial output
            rule_based = RuleBasedTransliterator()
            output = rule_based.transliterator(text=text)

            # Process output sentence by sentence
            paragraphs = output.split('\n')
            merged_para = []

            for paragraph in paragraphs:
                sentences = paragraph.split(".")
                merged_sentence = []

                for sentence in sentences:
                    try:
                        if sentence == "":
                            continue
                        # Get GPT response for level 0
                        gpt_response_1 = self.__get_gpt_response(text=sentence, level=0)
                        dictionary_1 = json.loads(gpt_response_1)

                        primary_list_item = 0  # Initialize the index for the while loop
                        dictionary_list = dictionary_1["word_list"]
                        dictionary_list = self.__remove_duplicates(dictionary_list)

                        while primary_list_item < len(dictionary_1["word_list"]):
                            for list_item in dictionary_list:

                                # Check for multiple occurrences of the same word
                                if sentence.count(list_item) > 1:
                                    sentence = sentence.replace(list_item, "<mask>", 1)
                                    dictionary_1["word_list"].append(list_item)
                                    dictionary_list.remove(list_item)
                                    dictionary_list.append(list_item)
                                    break

                                # Check for single occurrence of the word
                                elif sentence.count(list_item) == 1:
                                    sentence = sentence.replace(list_item, "<mask>", 1)
                                    break

                            primary_list_item += 1

                        merged_sentence.append(sentence)
                    except Exception as e:
                        print(f"Error processing sentence: {str(e)}")
                para = ". ".join(merged_sentence)
                merged_para.append(para)
            full_text = "\n".join(merged_para)
            return full_text
        except Exception as e:
            print(f"An error occurred in machine_mask: {str(e)}")
            return ""  # Return an empty string and an empty list in case of error

    def machine_suggest(self, text: str, changes: list) -> str:
        try:
            # Process output sentence by sentence
            paragraphs = text.split('\n')
            merged_para = []

            flag = 0
            for paragraph in paragraphs:
                sentences = paragraph.split(".")
                merged_sentence = []

                for sentence in sentences:
                    try:
                        if sentence == "":
                            continue
                        mask_count_in_sentence = sentence.count("<mask>")
                        sentence = sentence
                        mask_word = ""
                        for n in range(0, mask_count_in_sentence):
                            for m in range(n, mask_count_in_sentence):
                                ind = flag + m
                                if n == m:
                                    sentence = sentence.replace("<mask>", "####", 1)
                                    mask_word = changes[ind][1]
                                else:
                                    sentence = sentence.replace("<mask>", changes[ind][1], 1)

                            sentence = sentence.replace("####", "<mask>")
                            gpt_response = self.__get_gpt_response(text=sentence, level=1, word=mask_word)
                            dictionary = json.loads(gpt_response)
                            if mask_word in dictionary:
                                sentence = sentence.replace("<mask>", dictionary[mask_word])
                            else:
                                sentence = sentence.replace("<mask>", mask_word)

                        flag += mask_count_in_sentence
                        merged_sentence.append(sentence)
                    except Exception as e:
                        print(f"Error processing sentence: {str(e)}")
                para = ". ".join(merged_sentence)
                merged_para.append(para)
            full_text = "\n".join(merged_para)
            return full_text
        except Exception as e:
            print(f"An error occurred in machine_suggest: {str(e)}")
            return ""  # Return an empty string and an empty list in case of error
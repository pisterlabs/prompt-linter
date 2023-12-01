import openai
import os
import json

class vkeExtractor:
    def __init__(self) -> None:
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    def generate(self, task, prompt_id, vid, isExtracting = True):
        """
        task: match the prompt folder name
        prompt_id: prompt file name
        vid: video id
        isExtracting: if true, generate the output file
        """
        messages = [ {"role": "system", "content": "You are a intelligent assistant."} ]

        message = ""

        with open(f"./../LLM-prompts/{task}/{prompt_id}.txt") as prompt:
            message = prompt.read()


        with open("./../../metadata_subtitle_clean/yttemporal1b_train_0000of1024_clean.jsonl") as json_file:
            for line in json_file:
                data = json.loads(line)
                if data['id'] == vid:
                    message += data['subtitle']
                    break
        message += "\nLet's think step by step."

        messages.append(
            {"role": "user", "content": message},
        )
        model3 = "gpt-3.5-turbo-16k"
        model4 = "gpt-4"

        with open("./../LLM-scripts/current-prompt.txt", "w") as input_file:
            input_file.write(str(messages))
        print("input updated")

        if isExtracting:
            chat = openai.ChatCompletion.create(
                model=model4, messages=messages, temperature=0.0
            )
            reply = chat.choices[0].message.content
            output_dir = f"./../output/{task}-result/{prompt_id}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(f"{output_dir}/{vid}-{task}-{prompt_id}.txt", "w") as file:
                file.write(reply)
            print(vid + " generated")
    
    def verbExtractor(self):
        verb_list = set()
        i = 0
        verb_list_path = "./../output/cleaned/verb.txt"
        for vid in v_list:
            output_file_path = f"./../output/cleaned/1-COT-few-shot-distinguish-visual-quality-classification/{vid}.txt"
            input_file_path = f"./../output/Video-quality-classification-result/1-COT-few-shot-distinguish-visual-quality-classification/{vid}.txt"
            # Open the output file in write mode
            with open(output_file_path, "w") as output_file:
                with open(input_file_path, "r") as input_file:
                    # Loop through each line in the file
                    for line in input_file:
                        words = line.split()
                        if (len(words) > 3) and (words[0] == "Visual") and (words[1] == "Key") and (words[2] == "Event"):

                            nlp_line = nlp(line)
                            verbs = [token.text for token in nlp_line if token.pos_ == "VERB"]
                            if len(verbs) != 0:
                                verb_list.add(verbs[0])

    
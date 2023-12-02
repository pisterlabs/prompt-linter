from PyPDF2 import PdfReader
import json
import openai
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = ""

def main():
    # resume_path = "data/sample_resume.pdf"
    # resume_text = parse_pdf(resume_path)
    # resume_json = text_to_json(resume_text, "resume", "data/resume_template.json")
    # with open("data/sample_resume.json", "w") as f:
    #     json.dump(resume_json, f)
    
    job_description_path = "data/sample_jd.txt"
    job_description_text = open(job_description_path, "r").read()
    job_description_json = text_to_json(job_description_text, "job description", "data/jd_template.json")
    with open("data/sample_jd.json", "w") as f:
        json.dump(job_description_json, f)
    
    

def parse_pdf(path) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page.get_contents()
        print(page.extract_text())
        text = page.extract_text()
        text += text + "\n"
    return text

def text_to_json(text: str, text_type_name: str, template_path: str, max_attempts=3) -> dict:
    
    def attempt_conversion(prev_result):
        with open(template_path, "r") as f:
            template = f.read()
        content = f"Please convert the following {text_type_name} into a JSON object with exactly the following fields:" + "\n\n" + text + "\n\n" + template
        messages = [
            {"role": "user", "content": content},
        ]
        if prev_result is not None:
            messages.append({"role": "assistant", "content": prev_result})
            messages.append({"role": "user", "content": "No, that's not right. Please try again. Strictly follow the provided json template."})
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        return completion.choices[0].message.content
        
    completion = None
    success = False
    for i in range(max_attempts):
        try:
            completion = attempt_conversion(completion)
            output_json = json.loads(completion)
            template_json = json.load(open(template_path, "r"))
            for key in template_json.keys():
                if key not in output_json.keys():
                    raise ValueError(f"Key {key} not found in output JSON")
            success = True
            break
        except ValueError as e:
            print("JSON output does not match template")
            print(e)
    if not success:
        raise ValueError("Could not convert text to JSON")
    return output_json

   
if __name__ == "__main__":
    main()
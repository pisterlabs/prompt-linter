import openai
from utils import parse_results

def note_generator(chat_completion: openai.ChatCompletion, outline: dict[str, list[str]], level: str, subject: str):
    prompt = engineer_prompt_note_gen(outline, level, subject)
    model_output = chat_completion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system", "content":f"Your are a {level} {subject} notes generator. Generate notes in following format: \n Topic heading: \n \t Notes..."},
            # {"role": "system", "content":f"Your are a {level} {subject} notes generator."},
            {"role": "user", "content":prompt}
        ]
    )
    model_output = parse_results(model_output)
    return model_output
        
    
def engineer_prompt_note_gen(outline: dict[str, list[str]], level: str, subject: str):
    prompt = f'Generate quick revision notes for me on the following topics for a {subject} course on a {level} level. Include all equations. Explain each variable and concept. Do not say "sure, here are some notes..".\n\n'
    for topic in outline:
        prompt += f"- {topic}\n"
        if outline[topic] is not None:
            for subtopic in outline[topic]:
                prompt += f"    - {subtopic}\n"
    return prompt


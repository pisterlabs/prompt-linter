import openai
import os
from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPEN_API_KEY")

mental_illness = "Bulimia Nervosa"

def quote_generation(mental_illness):
    prompt = f'''Act as a world renowned Psychiatrist whose hobby is writing Motivational quotes in his free time for the patients.
    You know the mental illness of your patient and you have been regularly talking to them. To boost their morale, You write 10-15
    Motivational quotes(which are also related to their mental illness) for them daily so that their Mental health improves and
    they feel optimistic. The Mental Illness of the user is provided (delimited by <inp></inp). The output should be in a json file where the primary key is quote and it's key
    is a list of all the motivational quotes.

    <inp>
    Mental Illness: {mental_illness}
    </inp>

    OUTPUT FORMAT:
    '''

    scen = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": prompt},
        ],
        max_tokens=3000,
        temperature=0.4
        )

    motivation = scen['choices'][0]['message']['content']

    return motivation
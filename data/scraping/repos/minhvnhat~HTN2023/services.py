import cohere
from settings import COHERE_API

def generate_summary(moods: list):

    sorted_date_time = sorted(moods, key=lambda x: (x[0], -x[1]))

    DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    MOOD_VALS = ['Terrbile', 'Not Great', 'Neutral', 'Good', 'Awesome']
    
    mood_msg_rows = []
    for mood in sorted_date_time:
        msg = f"{DAY_NAMES[mood[0].weekday()]} - {'Day' if mood[1] else 'Night'}: {MOOD_VALS[mood[2]-1]}\n" 
        mood_msg_rows.append(msg)

    co = cohere.Client(COHERE_API)
    prompt = f"""
My mood in the week goes like this: 

"{''.join(mood_msg_rows)}"
Can you describe my general mood in the week, in only one sentence, with a cheerful tone?
    """

    print(prompt)

    response = co.chat(
        prompt, 
        model="command", 
        temperature=0.7
    )

    return response.text
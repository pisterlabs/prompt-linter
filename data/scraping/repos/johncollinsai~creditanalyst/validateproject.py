import openai

def validate_project(prompt, model, api_key):
    user_prompt = f"""Please confirm that {prompt} is a valid project description.
    Answer 'yes' or 'no'."""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.3,  
    )

    final_response = response.choices[0]["message"]["content"].strip().lower()

    if 'yes' in final_response or 'valid' in final_response:
        return True
    else:
        return False

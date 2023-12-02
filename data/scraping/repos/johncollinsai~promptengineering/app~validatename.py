import openai

def validate_company_name_gpt(prompt, modality, api_key):
    user_prompt = f"Please confirm that {prompt} is a valid company name by checking that it exists in a recognized corporate database or is listed on a stock exchange. Answer 'yes' or 'no'."

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"You are a {modality}."},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.3,  # Lower temperature to make validation more precise
    )

    final_response = response.choices[0]["message"]["content"].strip().lower()

    if 'yes' in final_response or 'valid' in final_response:
        return True
    else:
        return False

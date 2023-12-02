import openai

def validate_code(prompt, modality, api_key, max_seq_len):
    user_prompt = f"""Code is generally expected to follow certain syntactic and semantic rules of a programming language, whereas text or any other written form would not. Please determine if {prompt} is meant to be code by considering the following factors:
    Syntax Check: Does {prompt} follow the syntax rules of a specific programming language?
    Context: Is {prompt} embedded within other code or is it part of a code block?
    Purpose: Is the intent behind using {prompt} to execute some logic or operation?
    Environment: Is {prompt} being used in an environment specifically designed for code (e.g., an Integrated Development Environment or a Jupyter Notebook)?
    File Extension: If {prompt} is saved in a file, does the file have a code-specific extension (like .py, .js, .java, etc.)?
    Comments or Documentation: Does the surrounding context (comments, documentation, etc.) indicate that {prompt} is meant to be code?
    Answer 'yes' or 'no'."""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=max_seq_len,
        n=1,
        stop=None,
        temperature=0.3,  # Lower temperature to make validation more precise
    )

    final_response = response.choices[0]["message"]["content"].strip().lower()

    if 'yes' in final_response or 'valid' in final_response:
        return True
    else:
        return False